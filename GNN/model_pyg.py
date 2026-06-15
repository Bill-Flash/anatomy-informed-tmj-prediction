from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import (
        GATConv,
        GCNConv,
        GINConv,
        SAGEConv,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "未找到 torch_geometric。请先在当前环境安装 PyG（torch_geometric 及其依赖）。"
    ) from e


@dataclass(frozen=True)
class GraphSpecPyG:
    """固定 TMJ 小图结构（PyG 版）：节点数、global 节点索引、edge_index。"""

    n_nodes: int
    global_idx: int
    edge_index: torch.Tensor  # (2, E), long


def build_default_tmj_graph_spec_pyg(*, self_loops: bool = False) -> GraphSpecPyG:
    """构建与 `build_default_tmj_graph_spec()` 一致语义的固定 TMJ 小图（PyG edge_index）。

    节点顺序（n_nodes=13）与旧实现保持一致：
      0: global
      1: condyle_L
      2: condyle_R
      3: fossa_L
      4: fossa_R
      5: space_L
      6: space_R
      7: mandible_L
      8: mandible_R
      9:  global_demo
      10: global_occlusion
      11: global_mandible
      12: global_ratio
    """

    n_nodes = 13
    global_idx = 0

    edges: list[tuple[int, int]] = []

    def add_undirected(i: int, j: int) -> None:
        edges.append((i, j))
        edges.append((j, i))

    if self_loops:
        for i in range(n_nodes):
            edges.append((i, i))

    # global node connect to all
    for i in range(1, n_nodes):
        add_undirected(global_idx, i)

    # mirror edges
    add_undirected(1, 2)  # condyle L-R
    add_undirected(3, 4)  # fossa L-R
    add_undirected(5, 6)  # space L-R
    add_undirected(7, 8)  # mandible L-R

    # ipsilateral anatomy edges (L)
    add_undirected(1, 3)  # condyle_L - fossa_L
    add_undirected(3, 5)  # fossa_L - space_L
    add_undirected(1, 5)  # condyle_L - space_L
    add_undirected(7, 1)  # mandible_L - condyle_L

    # ipsilateral anatomy edges (R)
    add_undirected(2, 4)  # condyle_R - fossa_R
    add_undirected(4, 6)  # fossa_R - space_R
    add_undirected(2, 6)  # condyle_R - space_R
    add_undirected(8, 2)  # mandible_R - condyle_R

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    return GraphSpecPyG(n_nodes=n_nodes, global_idx=global_idx, edge_index=edge_index)


def _select_global_nodes(x: torch.Tensor, ptr: torch.Tensor, global_idx: int) -> torch.Tensor:
    """从 batch 后的节点表示中取出每张图的 global 节点表示。

    - x:   (total_nodes, hidden)
    - ptr: (num_graphs+1,)
    """
    # 每张图的起始 node offset + global_idx（通常为 0）
    idx = ptr[:-1] + int(global_idx)
    return x[idx]  # (B, hidden)


class TMJPYGNet(nn.Module):
    """TMJ 固定小图的图级二分类（PyG 版，输出 logit）。

    设计目标：
    - 结构简洁；
    - 保留“解剖图 + global 节点 readout”的直觉；
    - 便于在 GCN / GAT / GIN / SAGE 之间切换。
    """

    def __init__(
        self,
        *,
        num_features: int,
        cat_cardinalities: Optional[list[int]] = None,
        d_cat: int = 4,
        d_hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
        model_type: str = "gcn",  # gcn|gat|gin|sage
        gat_heads: int = 4,
        graph_spec: Optional[GraphSpecPyG] = None,
    ) -> None:
        super().__init__()

        self.graph_spec = graph_spec or build_default_tmj_graph_spec_pyg(self_loops=False)
        self.model_type = str(model_type).lower()
        self.dropout = float(dropout)

        self.num_encoder = nn.Linear(int(num_features), int(d_hidden))

        self.cat_cardinalities = cat_cardinalities or []
        if self.cat_cardinalities:
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(int(c), int(d_cat)) for c in self.cat_cardinalities]
            )
            self.cat_proj = nn.Linear(len(self.cat_cardinalities) * int(d_cat), int(d_hidden))
        else:
            self.cat_embeddings = None
            self.cat_proj = None

        self.convs = nn.ModuleList()
        hidden = int(d_hidden)
        layers = max(1, int(n_layers))

        if self.model_type == "gcn":
            for _ in range(layers):
                self.convs.append(GCNConv(hidden, hidden))
        elif self.model_type == "sage":
            for _ in range(layers):
                self.convs.append(SAGEConv(hidden, hidden))
        elif self.model_type == "gat":
            heads = max(1, int(gat_heads))
            # 输出维度保持 hidden：设置 concat=False（多头平均）
            for _ in range(layers):
                self.convs.append(GATConv(hidden, hidden, heads=heads, concat=False, dropout=self.dropout))
        elif self.model_type == "gin":
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden * 2),
                    nn.ReLU(),
                    nn.Linear(hidden * 2, hidden),
                )
                self.convs.append(GINConv(mlp))
        else:
            raise ValueError(f"Unknown model_type={model_type!r}, choose from gcn/gat/gin/sage")

        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

        # 缓存固定图 edge_index，便于外部复用/检查；训练时可直接用 data.edge_index
        self.register_buffer("default_edge_index", self.graph_spec.edge_index, persistent=False)

    def forward(self, data) -> torch.Tensor:
        # data.x: (total_nodes, num_features)
        # data.edge_index: (2, E_total) - 对固定结构图来说，批处理时 PyG 会自动偏移 index
        x = self.num_encoder(data.x)

        # 类别特征注入到 global 节点（每张图一个 global）
        if self.cat_embeddings is not None and hasattr(data, "cat") and data.cat is not None:
            # 约定：data.cat shape = (B, n_cat)
            cat = data.cat
            if cat.dim() == 1:
                cat = cat.view(-1, len(self.cat_cardinalities))
            emb_list = [emb(cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_vec = torch.cat(emb_list, dim=1)  # (B, n_cat*d_cat)
            cat_h = self.cat_proj(cat_vec)  # (B, hidden)

            if hasattr(data, "ptr") and data.ptr is not None:
                g_idx = data.ptr[:-1] + int(self.graph_spec.global_idx)
            else:
                # fallback：若没有 ptr，就假设每张图节点数固定为 n_nodes
                n_nodes = int(self.graph_spec.n_nodes)
                batch_size = int(cat_h.shape[0])
                g_idx = torch.arange(batch_size, device=x.device, dtype=torch.long) * n_nodes + int(
                    self.graph_spec.global_idx
                )
            x[g_idx] = x[g_idx] + cat_h

        # message passing
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.out_norm(x)

        # readout：取 global 节点（保持解剖直觉）
        if hasattr(data, "ptr") and data.ptr is not None:
            g = _select_global_nodes(x, data.ptr, int(self.graph_spec.global_idx))
        else:
            # fallback：均值池化（理论上不会走到这里）
            from torch_geometric.nn import global_mean_pool

            g = global_mean_pool(x, data.batch)

        logits = self.head(g).squeeze(-1)  # (B,)
        return logits


__all__ = [
    "GraphSpecPyG",
    "build_default_tmj_graph_spec_pyg",
    "TMJPYGNet",
]


