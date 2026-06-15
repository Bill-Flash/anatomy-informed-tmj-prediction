"""
数据加载与预处理模块。

目标：
- 从项目根目录的 data.xlsx 读取数据；
- 复用 notebook 中的主要清洗思路，但以脚本形式实现一个“可运行的最小版本”；
- 输出适合喂给深度学习模型的:
    - X_num: np.ndarray[float32], shape (N, D)
    - y: np.ndarray[int64],      shape (N,)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data.xlsx"


def _clean_header_level(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower().startswith("unnamed"):
        return ""
    return text


def _flatten_columns(multi_columns: pd.MultiIndex) -> list[str]:
    flattened: list[str] = []
    for idx, col_tuple in enumerate(multi_columns):
        parts = []
        for level in col_tuple:
            cleaned = _clean_header_level(level)
            if cleaned:
                parts.append(cleaned)
        flattened.append("_".join(parts) if parts else f"col_{idx}")
    return flattened


def _fix_left_columns_after_flatten(columns: list[str]) -> list[str]:
    """修复扁平化后出现的“左列名丢失”问题。

    典型输入列序列形态（来自当前 data.xlsx）：
        <指标/mm>_右, 左, <指标>   ->  <指标/mm>_右, <指标/mm>_左

    说明：
    - 这里只做非常保守的推断：仅当中间列名恰为“左”，且前一列以“_右”结尾、
      后一列不带“_左/_右”后缀时，才将后一列改名为推断出的“_左”列名。
    - 中间那些“左”列通常是残留的表头/占位列，不应进入特征，因此后续会统一删除。
    """
    cols = list(columns)
    for i in range(1, len(cols) - 1):
        prev, cur, nxt = cols[i - 1], cols[i], cols[i + 1]
        if cur != "左":
            continue
        if not isinstance(prev, str) or not prev.endswith("_右"):
            continue
        if not isinstance(nxt, str) or (not nxt) or nxt in {"左", "右"}:
            continue
        if nxt.endswith("_左") or nxt.endswith("_右"):
            continue

        inferred_left = f"{prev[:-2]}_左"
        cols[i + 1] = inferred_left

    return cols


def _dedup_column_names(names: list[str]) -> list[str]:
    """保证列名唯一：对重复列名追加稳定后缀（__dup2, __dup3, ...）。"""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        key = str(n)
        cnt = seen.get(key, 0) + 1
        seen[key] = cnt
        out.append(key if cnt == 1 else f"{key}__dup{cnt}")
    return out


def _normalize_single_row_columns(columns: list[object]) -> list[str]:
    """规范单行表头，并尽量恢复左右成对测量列名。

    有些导出的 Excel 已经把两行表头压平成一行，左右成对列常表现为：
        指标/mm, Unnamed: n, 指标均值

    对这类序列，推断为：
        指标/mm_右, 指标/mm_左, 指标均值

    这样后续 GNN 的解剖节点映射仍能识别 `_左` / `_右`。
    """
    raw_cols: list[str] = []
    for idx, value in enumerate(columns):
        text = _clean_header_level(value)
        raw_cols.append(text if text else f"col_{idx}")

    cols = list(raw_cols)
    for i in range(len(cols) - 2):
        cur, nxt, nxt2 = cols[i], cols[i + 1], cols[i + 2]
        if cur.startswith("col_") or not nxt.startswith("col_") or nxt2.startswith("col_"):
            continue
        if cur.endswith("_右") or cur.endswith("_左"):
            continue

        base = cur
        for suffix in ("/mm", "/°", "（0，前位；1，中位；后位）"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        if base and (nxt2 == base or nxt2.startswith(base)):
            cols[i] = f"{cur}_右"
            cols[i + 1] = f"{cur}_左"

    for i in range(len(cols) - 1):
        cur, nxt = cols[i], cols[i + 1]
        if "髁突与关节窝的相对位置" in cur and nxt.startswith("col_"):
            cols[i] = f"{cur}_右"
            cols[i + 1] = f"{cur}_左"

    return _dedup_column_names(cols)


_MISSING_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "-",
    "--",
    "—",
    "－",
    "无",
    "缺失",
    "未知",
}


def _coerce_numeric_like_object_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: set[str] | None = None,
    min_parse_rate: float = 0.8,
) -> pd.DataFrame:
    """将“看起来是数值”的 object 列尽量转成 float。

    规则：
    - 对 object/string 列进行字符串清洗（去空格、常见缺失标记）；
    - 从字符串中提取第一个数字（支持负号、小数、科学计数）；
    - 若在“原本非缺失”的样本中解析成功比例 >= min_parse_rate，则认为该列应为数值列并替换为 float。
    - 解析失败的值视为未知，统一置为 NaN，后续由数值缺失填补逻辑处理。
    """
    exclude_cols = exclude_cols or set()
    obj_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and (df[c].dtype == "object" or str(df[c].dtype) == "string")
    ]
    if not obj_cols:
        return df

    df = df.copy()
    num_pat = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    for c in obj_cols:
        s = df[c]
        non_missing_mask = ~pd.isna(s)
        if non_missing_mask.sum() == 0:
            continue

        # 统一按字符串处理
        s_str = s.astype("string").str.strip()
        # 常见缺失标记 -> NaN
        s_norm = s_str.str.lower()
        s_str = s_str.mask(s_norm.isin(_MISSING_TOKENS))

        # 去掉常见分隔符/单位噪声：先提取数字子串，再转数值
        extracted = s_str.str.replace(",", "", regex=False).str.extract(f"({num_pat})", expand=False)
        s_num = pd.to_numeric(extracted, errors="coerce")

        # 只在原本非缺失的样本里评估解析成功率
        parse_rate = float(s_num[non_missing_mask].notna().mean())
        if parse_rate >= min_parse_rate:
            df[c] = s_num.astype("float64")

    return df


def _impute_left_right_symmetric_columns(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """对左右对称的数值列做“对侧补全 + 双缺失用 pair median 填补”。

    适用场景：
    - 列名以 `_左` / `_右` 结尾（例如 `关节窝深度/mm_左`、`关节窝深度/mm_右`）
    - 当一侧为 NaN、对侧有值时，用对侧值补全

    注意：
    - 仅作用于数值 DataFrame（避免 object 列导致 dtype 污染）
    - 若两侧均缺失：用该“左右对称指标”的总体中位数填补（left/right 合并后的 median）
      之后其它仍缺失的列再由全局缺失填补策略兜底
    """
    if numeric_df.empty:
        return numeric_df

    df = numeric_df.copy()
    left_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_左")]

    for left in left_cols:
        base = left[:-2]  # 去掉末尾“_左”
        right = f"{base}_右"
        if right not in df.columns:
            continue

        # 左缺用右补，右缺用左补
        df[left] = df[left].fillna(df[right])
        df[right] = df[right].fillna(df[left])

        # 若左右仍同时缺失，则用该指标的总体中位数填补（合并左右列后取 median）
        both_missing = df[left].isna() & df[right].isna()
        if bool(both_missing.any()):
            pair_median = float(pd.concat([df[left], df[right]], axis=0).median(skipna=True))
            # 如果整列都是 NaN，pair_median 会是 nan，此时保持 NaN，交给后续兜底
            if not np.isnan(pair_median):
                df.loc[both_missing, left] = pair_median
                df.loc[both_missing, right] = pair_median

    return df


def load_raw_dataframe() -> pd.DataFrame:
    """从 data.xlsx 读取主工作表（'总' 或第一张表），并展平多级表头。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    xl = pd.ExcelFile(DATA_PATH)
    sheet_names = xl.sheet_names
    target_sheet = "总" if "总" in sheet_names else sheet_names[0]

    df_raw = xl.parse(target_sheet, header=[0, 1])
    df = df_raw.copy()
    # 1) 展平表头（两行）
    df.columns = _flatten_columns(df_raw.columns)
    # 2) 修复“左侧列名丢失 -> 扁平后出现大量重复 '左' ”的情况
    df.columns = _fix_left_columns_after_flatten(list(df.columns))
    # 3) 删除残留的“左”占位列（它们不是有效特征列名）
    if "左" in set(df.columns):
        df = df.drop(columns=[c for c in df.columns if c == "左"])
    # 4) 兜底：确保列名唯一
    df.columns = _dedup_column_names(list(df.columns))

    if "关节紊乱" not in df.columns:
        # 兼容单行表头导出：若两行表头会把首个数据行误读成第二级表头，
        # 标签列将变成“关节紊乱_0”等，此时回退到 header=0。
        df = xl.parse(target_sheet, header=0)
        df.columns = _normalize_single_row_columns(list(df.columns))

    return df


def _build_features_from_dataframe(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """在给定 DataFrame 上执行清洗 + 特征工程，构建特征与标签。"""
    # 1. 删除隐私列与明显无关列
    ignore_cols = ["姓名", "影像", "关节症状", "牙型", "骨型"]
    df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    # 2. 删除缺失率过高的列
    missing_profile = df.isna().mean()
    high_missing_cols = missing_profile[missing_profile > 0.5].index.tolist()
    df = df.drop(columns=high_missing_cols)

    # 3. 删除关键字段缺失行（如果存在）
    required_cols = [c for c in ["身份证号", "联系电话"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)
        # 这些字段仅用于筛掉明显不合格样本，不应进入模型特征
        df = df.drop(columns=required_cols)

    # 4. 目标列：关节紊乱
    label_col = "关节紊乱"
    if label_col not in df.columns:
        raise KeyError("数据中未找到标签列 '关节紊乱'，请检查表头。")

    y_numeric = pd.to_numeric(df[label_col], errors="coerce")
    valid_label_mask = y_numeric.notna()
    if not bool(valid_label_mask.all()):
        df = df.loc[valid_label_mask].copy()
        y_numeric = y_numeric.loc[valid_label_mask]
    y = y_numeric.astype(int)

    # 5. 指定“应作为类别特征”的列名（若存在则启用）
    cat_cols = [
        "性别",
        "牙型",
        "骨型",
        "面型",
        "髁突与关节窝的相对位置_右",
        "髁突与关节窝的相对位置_左",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df_cat = pd.DataFrame(index=df.index)
    cat_cardinalities: List[int] = []

    for c in cat_cols:
        # 特殊处理：髁突与关节窝的相对位置（0/1/2）压缩为“中位 vs 非中位”二分类
        if c in {"髁突与关节窝的相对位置_右", "髁突与关节窝的相对位置_左"}:
            raw = df[c]
            # 约定：1 表示中位，其余（0/2 等）视为非中位；缺失保持为 NaN，稍后统一填补
            mapped = raw.map(
                lambda v: "中位" if v == 1 else ("非中位" if pd.notna(v) else np.nan)
            )
            series = mapped.astype("string").fillna("缺失")
        else:
            # 统一当作字符串类别处理；缺失值单独作为一个类别
            series = df[c].astype("string").fillna("缺失")

        cat = series.astype("category")
        df_cat[c] = cat.cat.codes
        cat_cardinalities.append(len(cat.cat.categories))

    if cat_cols:
        X_cat = df_cat[cat_cols].to_numpy(dtype="int64")
    else:
        # 若当前数据中不存在任何类别特征，返回一个 (N, 0) 的占位数组
        X_cat = np.zeros((len(df), 0), dtype="int64")

    # 5.5 将“应为数值但被读成 object”的列尽量转成数值列（未知值 -> NaN）
    # 注意：不要动标签列与明确的类别列
    df = _coerce_numeric_like_object_columns(
        df,
        exclude_cols=set([label_col, *cat_cols]),
        min_parse_rate=0.8,
    )

    # 6. 数值特征选择：保留所有数值列，去掉标签列本身以及已经当作类别处理的列
    numeric_df = df.select_dtypes(include="number").copy()
    drop_cols = [label_col] + [c for c in cat_cols if c in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=[c for c in drop_cols if c in numeric_df.columns])

    # 6.0 对称补全：若左/右对应列一侧缺失，优先用对侧值补全（比全局中位数更贴合解剖对称性）
    numeric_df = _impute_left_right_symmetric_columns(numeric_df)

    # 缺失值用中位数填充
    numeric_df = numeric_df.fillna(numeric_df.median())

    # 6.a 针对右偏长尾特征做 log1p 变换（先压尾再标准化）
    # 条件：偏度 > 1 且最小值 >= 0，避免对称/已居中或有负值的列被错误变换
    skewness = numeric_df.skew(numeric_only=True)
    long_tail_cols: List[str] = []
    for col in numeric_df.columns:
        if skewness.get(col, 0.0) > 1.0 and numeric_df[col].min() >= 0:
            long_tail_cols.append(col)
    if long_tail_cols:
        numeric_df[long_tail_cols] = np.log1p(numeric_df[long_tail_cols])

    # 6.b 删除方差极低 / 近乎常数的数值列（对模型贡献有限，易导致数值不稳定）
    low_var_cols: List[str] = []
    for col in numeric_df.columns:
        series = numeric_df[col]
        # 只有一个取值或极小波动的列
        if series.nunique(dropna=True) <= 1:
            low_var_cols.append(col)
        else:
            value_range = float(series.max() - series.min())
            mean_abs = float(series.abs().mean()) + 1e-6
            if value_range / mean_abs < 0.01:  # 相对波动 < 1%
                low_var_cols.append(col)
    if low_var_cols:
        numeric_df = numeric_df.drop(columns=low_var_cols)

    num_feature_names = list(numeric_df.columns)
    X_num = numeric_df.astype("float32").to_numpy()
    y_arr = y.to_numpy(dtype="int64")

    # 可选调试输出：用于核对进入模型的列名（避免 sweep 时刷屏，默认关闭）
    if os.environ.get("CRK_PRINT_FEATURES", "").strip() == "1":
        print("=== Feature debug (CRK_PRINT_FEATURES=1) ===")
        print("X_num shape:", X_num.shape)
        print("X_cat shape:", X_cat.shape)
        print("label_col:", label_col)
        print("cat_cols:", cat_cols)
        print("num_feature_names:")
        for i, n in enumerate(num_feature_names, 1):
            print(f"{i:03d}\t{n}")

    return X_num, X_cat, y_arr, cat_cardinalities, num_feature_names


def prepare_features_and_labels() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """构建一个与 notebook 逻辑尽量一致、但更简洁的清洗 + 特征选择流程（监督分类用）。"""
    df = load_raw_dataframe()
    return _build_features_from_dataframe(df)


def _build_real_and_virtual_dataframe(
    virtual_csv_path: str | Path,
) -> tuple[pd.DataFrame, int]:
    """加载真实 + 虚拟数据，并在列维度对齐后纵向拼接。

    返回:
        df_all:   合并后的 DataFrame
        n_real:   其中前 n_real 行为真实数据，其余为虚拟数据
    """
    df_real = load_raw_dataframe()
    virtual_csv_path = Path(virtual_csv_path)
    if not virtual_csv_path.exists():
        raise FileNotFoundError(f"未找到虚拟数据文件: {virtual_csv_path}")

    df_virtual = pd.read_csv(virtual_csv_path)
    n_real = len(df_real)

    # 对齐列名：取并集，并为缺失列填充 NaN，确保两侧列顺序一致
    all_cols = sorted(set(df_real.columns).union(df_virtual.columns))

    for df in (df_real, df_virtual):
        missing_cols = [c for c in all_cols if c not in df.columns]
        for c in missing_cols:
            df[c] = np.nan

        df = df  # 仅为类型提示友好，无实质作用

    df_real = df_real[all_cols]
    df_virtual = df_virtual[all_cols]

    df_all = pd.concat([df_real, df_virtual], axis=0, ignore_index=True)
    return df_all, n_real


def prepare_pretraining_features(
    virtual_csv_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """为自监督预训练阶段准备特征矩阵。

    - 若 virtual_csv_path 为 None：仅使用真实 Excel 数据；
    - 若提供虚拟数据 CSV：将真实数据与虚拟数据按列对齐后纵向拼接，再进行统一清洗与特征工程。
    """
    if virtual_csv_path is None:
        df_real = load_raw_dataframe()
        X_num, X_cat, _y, cat_cardinalities, num_feature_names = _build_features_from_dataframe(
            df_real
        )
        return X_num, X_cat, cat_cardinalities, num_feature_names

    df_all, _ = _build_real_and_virtual_dataframe(virtual_csv_path)
    X_num, X_cat, _y, cat_cardinalities, num_feature_names = _build_features_from_dataframe(df_all)
    return X_num, X_cat, cat_cardinalities, num_feature_names


def prepare_features_and_labels_with_virtual(
    virtual_csv_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """在“真实 + 虚拟”联合特征空间下，仅返回真实样本的特征与标签。

    用途：
        当下游训练需要加载在“真实+虚拟数据”上预训练得到的权重时，
        为保证数值特征维度与顺序完全一致，监督训练也需使用同一套特征工程。
    """
    df_all, n_real = _build_real_and_virtual_dataframe(virtual_csv_path)
    X_num_all, X_cat_all, y_all, cat_cardinalities, num_feature_names = _build_features_from_dataframe(
        df_all
    )

    X_num_real = X_num_all[:n_real]
    X_cat_real = X_cat_all[:n_real]
    y_real = y_all[:n_real]

    return X_num_real, X_cat_real, y_real, cat_cardinalities, num_feature_names


if __name__ == "__main__":
    X_num, X_cat, y, cat_cardinalities, num_feature_names = prepare_features_and_labels()
    print("=== 监督分类数据 ===")
    print("X_num shape:", X_num.shape)
    print("X_cat shape:", X_cat.shape)
    print("y shape:", y.shape)
    print("cat_cardinalities:", cat_cardinalities)
    print("num_feature_names (前 10 列):", num_feature_names[:])
