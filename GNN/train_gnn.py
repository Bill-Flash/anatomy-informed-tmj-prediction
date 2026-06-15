from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset as TorchDataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except Exception as e:  # pragma: no cover
    raise ImportError(
        "未找到 torch_geometric。你已要求使用 PyG 版本，请先安装 torch_geometric。"
    ) from e

# 确保从项目根目录运行 `python3 GNN/train_gnn.py` 时也能 import 到同级目录下的 `DL/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DL.data import prepare_features_and_labels  # noqa: E402
from GNN.model_pyg import TMJPYGNet, build_default_tmj_graph_spec_pyg  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """尽量固定训练随机性，使同一环境下多次运行结果更可复现。

    说明：
    - `PYTHONHASHSEED` 严格来说应在 Python 启动前设置；这里仍设置一次作为兜底。
    - CUDA 的完全确定性还受驱动/算子影响；这里开启 PyTorch 的常用确定性开关。
    """
    seed = int(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # cuBLAS 确定性（需在 CUDA context 创建前设置更稳；这里尽量早设置）
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 更强的确定性约束：若遇到不支持的算子可能抛错
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


@dataclass(frozen=True)
class NodeIndex:
    global_idx: int = 0
    condyle_l: int = 1
    condyle_r: int = 2
    fossa_l: int = 3
    fossa_r: int = 4
    space_l: int = 5
    space_r: int = 6
    mandible_l: int = 7
    mandible_r: int = 8
    global_demo: int = 9
    global_occlusion: int = 10
    global_mandible: int = 11
    global_ratio: int = 12


NODE_INDEX = NodeIndex()


def _side_from_name(name: str) -> Optional[str]:
    if name.endswith("_左"):
        return "L"
    if name.endswith("_右"):
        return "R"
    return None


def _assign_node_for_feature(name: str) -> int:
    """把一个特征列名分配到某个解剖节点。

    规则是“尽量合理 + 可解释”，不是唯一正确答案：
    - 关节窝* -> fossa
    - *间隙* -> space
    - 髁突* 且有左右 -> condyle
    - 下颌体长度/下颌支长度 且有左右 -> mandible
    - 年龄（无左右）-> global_demo
    - 咬合/角度/垂直维度类（无左右）-> global_occlusion
    - 下颌整体尺度类（无左右）-> global_mandible
    - 比例/指数类（无左右）-> global_ratio
    - 其它 -> global
    """
    side = _side_from_name(name)
    base = name[:-2] if side is not None else name

    # ---- 全局语义（无左右后缀）优先分流到对应 global-type 节点 ----
    if side is None:
        if "年龄" in base:
            return NODE_INDEX.global_demo

        # 咬合/角度/垂直维度：通常是个体整体姿态/咬合模式，不绑定某一侧解剖节点
        if (
            ("切导角" in base)
            or ("咬合平面角" in base)
            or ("下颌平面角" in base)
            or ("髁突到咬合面的垂直高度" in base)
        ):
            return NODE_INDEX.global_occlusion

        # 下颌整体尺度（无左右）
        if ("下颌体长度" in base) or ("下颌支长度" in base) or ("下颌骨长度" in base):
            return NODE_INDEX.global_mandible

        # 比例/指数
        if ("比例" in base) or ("CHO" in base) or ("α" in base):
            return NODE_INDEX.global_ratio

    if "关节窝" in base and side is not None:
        return NODE_INDEX.fossa_l if side == "L" else NODE_INDEX.fossa_r
    if "间隙" in base and side is not None:
        return NODE_INDEX.space_l if side == "L" else NODE_INDEX.space_r
    if "髁突" in base and side is not None:
        return NODE_INDEX.condyle_l if side == "L" else NODE_INDEX.condyle_r
    if ("下颌体长度" in base or "下颌支长度" in base) and side is not None:
        return NODE_INDEX.mandible_l if side == "L" else NODE_INDEX.mandible_r

    return NODE_INDEX.global_idx


def build_node_feature_tensor(
    x_num: np.ndarray,
    num_feature_names: List[str],
    *,
    n_nodes: int,
) -> np.ndarray:
    """把 (N, F) 的表格数值特征变成 (N, n_nodes, F) 的节点特征张量。

    这里采用“全特征维度稀疏分配”的方式：
    - 每个节点的输入是长度为 F 的向量
    - 属于该节点的特征位置填入数值，其它位置为 0
    """
    n_samples, n_features = x_num.shape
    out = np.zeros((n_samples, n_nodes, n_features), dtype=np.float32)

    for j, name in enumerate(num_feature_names):
        node_id = _assign_node_for_feature(name)
        out[:, node_id, j] = x_num[:, j]

    return out


class GraphTabularDatasetPyG(TorchDataset):
    """把 (N, n_nodes, F) 的节点特征张量包装成 PyG Data。

    说明：
    - 固定图结构：edge_index 对所有样本相同；
    - 每个样本的图都有 13 个节点（由 graph_spec 决定）；
    - 类别特征 cat 以 shape=(1, n_cat) 存入，batch 后得到 (B, n_cat)。
    """

    def __init__(
        self,
        x_nodes_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        *,
        graph_spec,
    ) -> None:
        self.x_nodes_num = x_nodes_num.astype(np.float32, copy=False)
        self.x_cat = x_cat.astype(np.int64, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.graph_spec = graph_spec

        # 固定 edge_index（PyG batch 时会自动做 index 偏移）
        self.edge_index = self.graph_spec.edge_index.clone().detach()

        n_nodes = int(self.graph_spec.n_nodes)
        if self.x_nodes_num.ndim != 3 or int(self.x_nodes_num.shape[1]) != n_nodes:
            raise ValueError(
                f"x_nodes_num 期望 shape=(N,{n_nodes},F)，但拿到 {self.x_nodes_num.shape}"
            )
        if self.x_cat.ndim != 2:
            raise ValueError(f"x_cat 期望 shape=(N,C)，但拿到 {self.x_cat.shape}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Data:
        x = torch.tensor(self.x_nodes_num[idx], dtype=torch.float32)  # (n_nodes, F)
        y = torch.tensor([self.y[idx]], dtype=torch.float32)  # (1,)
        cat = torch.tensor(self.x_cat[idx : idx + 1], dtype=torch.long)  # (1, C)
        return Data(x=x, edge_index=self.edge_index, y=y, cat=cat)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    threshold: Optional[float] = None,
    threshold_strategy: str = "acc",
) -> Dict[str, float | int]:
    model.eval()
    y_true = []
    y_score = []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_score.append(probs)
        y_true.append(batch.y.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0) if y_true else np.zeros((0,), dtype=np.float32)
    y_score_np = np.concatenate(y_score, axis=0) if y_score else np.zeros((0,), dtype=np.float32)
    if y_true_np.size == 0:
        return {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "threshold": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    # 统一 shape：模型输出常见为 (N, 1)，转成 (N,)
    y_true_np = y_true_np.reshape(-1).astype(np.int64)
    y_score_np = y_score_np.reshape(-1).astype(np.float32)

    try:
        auc = float(roc_auc_score(y_true_np, y_score_np))
    except Exception:
        auc = float("nan")

    try:
        pr = float(average_precision_score(y_true_np, y_score_np))
    except Exception:
        pr = float("nan")

    # 阈值选择：若 threshold is None，则在当前评估数据上搜索使目标最大（默认 max ACC）
    if threshold is None:
        unique_scores = np.unique(y_score_np)
        # 让“全预测为 0 / 全预测为 1”也成为候选
        thresholds = np.concatenate(([-1e-6], unique_scores, [1.0 + 1e-6])).astype(np.float32)

        # shape: (T, N)
        preds = (y_score_np[None, :] >= thresholds[:, None]).astype(np.int64)
        # accuracy per threshold
        accs = (preds == y_true_np[None, :]).mean(axis=1)

        if threshold_strategy.lower() == "acc":
            scores = accs
        elif threshold_strategy.lower() == "f1":
            # 备用：按 F1 选阈值（逐阈值计算，阈值数通常不大；如需优化可再向量化）
            scores = np.asarray(
                [
                    f1_score(y_true_np, preds[i], pos_label=1, zero_division=0)
                    for i in range(preds.shape[0])
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown threshold_strategy={threshold_strategy!r}, use 'acc' or 'f1'.")

        best = np.nanmax(scores)
        # tie-break：优先选离 0.5 最近的阈值，保证稳定性
        candidate_idxs = np.where(scores == best)[0]
        if candidate_idxs.size == 0:
            best_threshold = 0.5
        else:
            best_idx = candidate_idxs[np.argmin(np.abs(thresholds[candidate_idxs] - 0.5))]
            best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = float(threshold)

    # 基于 best_threshold 计算阈值类指标 + confusion matrix
    y_pred = (y_score_np >= best_threshold).astype(np.int64)
    try:
        acc = float(accuracy_score(y_true_np, y_pred))
    except Exception:
        acc = float("nan")

    try:
        precision = float(precision_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    except Exception:
        precision = float("nan")

    try:
        recall = float(recall_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    except Exception:
        recall = float("nan")

    try:
        f1 = float(f1_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    except Exception:
        f1 = float("nan")

    try:
        cm = confusion_matrix(y_true_np, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    except Exception:
        tn, fp, fn, tp = (0, 0, 0, 0)

    return {
        "auc": auc,
        "pr_auc": pr,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        y = batch.y.view(-1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / max(total, 1)


def run_kfold(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.ndarray,
    num_feature_names: List[str],
    cat_cardinalities: List[int],
    *,
    seed: int,
    n_splits: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    d_hidden: int,
    n_layers: int,
    dropout: float,
    model_type: str,
    gat_heads: int,
) -> None:
    graph_spec = build_default_tmj_graph_spec_pyg(self_loops=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    fold_best_aucs: list[float] = []
    fold_best_accs: list[float] = []
    fold_best_f1s: list[float] = []
    fold_best_thresholds: list[float] = []

    print(f"使用 {n_splits} 折交叉验证训练 PyG-GNN（model={model_type}）...")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num, y), start=1):
        print(f"\n===== Fold {fold}/{n_splits} =====")
        # 每折固定一个独立 seed，避免“前面 fold 的随机数消费”影响后面 fold
        set_global_seed(int(seed) + int(fold), deterministic=True)
        Xn_tr_raw, Xn_va_raw = X_num[tr_idx], X_num[va_idx]
        Xc_tr, Xc_va = X_cat[tr_idx], X_cat[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 标准化：只在训练折拟合
        scaler = StandardScaler()
        Xn_tr = scaler.fit_transform(Xn_tr_raw).astype(np.float32)
        Xn_va = scaler.transform(Xn_va_raw).astype(np.float32)

        # 构图：每个样本一张固定结构的小图
        xg_tr = build_node_feature_tensor(Xn_tr, num_feature_names, n_nodes=graph_spec.n_nodes)
        xg_va = build_node_feature_tensor(Xn_va, num_feature_names, n_nodes=graph_spec.n_nodes)

        tr_ds = GraphTabularDatasetPyG(xg_tr, Xc_tr, y_tr.astype(np.float32), graph_spec=graph_spec)
        va_ds = GraphTabularDatasetPyG(xg_va, Xc_va, y_va.astype(np.float32), graph_spec=graph_spec)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

        model = TMJPYGNet(
            num_features=X_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            d_hidden=d_hidden,
            n_layers=n_layers,
            dropout=dropout,
            graph_spec=graph_spec,
            model_type=model_type,
            gat_heads=gat_heads,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = float("-inf")
        best_state: Optional[dict] = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
            va_m = evaluate_metrics(model, va_loader, threshold=None, threshold_strategy="acc")

            print(
                f"Fold {fold} | Epoch {epoch:02d} | train_loss={tr_loss:.4f} "
                f"val_auc={va_m['auc']:.4f} val_pr_auc={va_m['pr_auc']:.4f} "
                f"val_acc={va_m['acc']:.4f} val_f1={va_m['f1']:.4f} thr={va_m['threshold']:.4f}"
            )

            if not np.isnan(float(va_m["auc"])) and float(va_m["auc"]) > best_auc:
                best_auc = float(va_m["auc"])
                best_state = deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(
                    f"Fold {fold}: Early stopping at epoch {epoch} "
                    f"(patience={patience}, best_val_auc={best_auc:.4f})"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        best_m = evaluate_metrics(model, va_loader, threshold=None, threshold_strategy="acc")
        fold_best_aucs.append(float(best_m["auc"]))
        fold_best_accs.append(float(best_m["acc"]))
        fold_best_f1s.append(float(best_m["f1"]))
        fold_best_thresholds.append(float(best_m["threshold"]))
        print(
            f"Fold {fold} best checkpoint | val_auc={best_m['auc']:.4f} "
            f"val_pr_auc={best_m['pr_auc']:.4f} val_acc={best_m['acc']:.4f} "
            f"val_f1={best_m['f1']:.4f} thr={best_m['threshold']:.4f} "
            f"cm=[[{best_m['tn']}, {best_m['fp']}], [{best_m['fn']}, {best_m['tp']}]]"
        )

    arr = np.asarray(fold_best_aucs, dtype=float)
    arr_acc = np.asarray(fold_best_accs, dtype=float)
    arr_f1 = np.asarray(fold_best_f1s, dtype=float)
    arr_thr = np.asarray(fold_best_thresholds, dtype=float)
    print("\n===== 总结（以 ROC-AUC 为主） =====")
    print("各折 best ROC-AUC:", [f"{v:.4f}" for v in arr])
    print("平均 ROC-AUC:", float(np.nanmean(arr)))
    print("ROC-AUC std:", float(np.nanstd(arr)))
    print("各折 best ACC(thr自适应):", [f"{v:.4f}" for v in arr_acc])
    print("平均 ACC:", float(np.nanmean(arr_acc)))
    print("ACC std:", float(np.nanstd(arr_acc)))
    print("各折 best F1(在thr下):", [f"{v:.4f}" for v in arr_f1])
    print("平均 F1:", float(np.nanmean(arr_f1)))
    print("F1 std:", float(np.nanstd(arr_f1)))
    print("各折 thr:", [f"{v:.4f}" for v in arr_thr])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GIN/GNN training for TMJ tabular-to-graph classification.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--model",
        type=str,
        default="gin",
        choices=["gcn", "gat", "gin", "sage"],
        help="PyG 模型类型：gcn/gat/gin/sage",
    )
    p.add_argument("--gat-heads", type=int, default=4, help="GAT 多头数（仅 --model gat 生效）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 全局先设一次 seed（fold 内会再设一次 fold-specific seed）
    set_global_seed(args.seed, deterministic=True)

    X_num, X_cat, y, cat_cardinalities, num_feature_names = prepare_features_and_labels()
    # 二分类，y 用 float32（0/1）
    y = y.astype(np.float32)

    run_kfold(
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        num_feature_names=num_feature_names,
        cat_cardinalities=cat_cardinalities,
        seed=args.seed,
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        model_type=args.model,
        gat_heads=args.gat_heads,
    )


if __name__ == "__main__":
    main()

