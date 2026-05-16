from pathlib import Path, PureWindowsPath
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# =========================
# RAIR 特征表候选路径（按优先级）
# 1) 新版严谨命名：rair_joined.csv
# 2) 新版患者级表：rair_feature_table.csv
# 3) 旧版兼容表：rair_attn8_joined_strictdose.csv
# =========================
RAIR_FEATURES_CSV_CANDIDATES = [
    DATA_DIR / "rair" / "rair_joined.csv",
    DATA_DIR / "rair" / "rair_feature_table.csv",
    DATA_DIR / "rair" / "rair_attn8_joined_strictdose.csv",
]

# =========================
# RAIR 索引表候选路径（按优先级）
# 1) 新版：rair_index.csv
# 2) 旧版：rair_index_1.csv
# =========================
RAIR_INDEX_CSV_CANDIDATES = [
    DATA_DIR / "rair" / "rair_index.csv",
    DATA_DIR / "rair" / "rair_index_1.csv",
]

RAIR_NPZ_DIR = DATA_DIR / "npz"


def normalize_pid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s or None


def _safe_scalar(x):
    """
    把 numpy scalar / pandas scalar 转成 python 原生类型
    """
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, np.generic):
        return x.item()
    return x


def _pick_existing_features_csv():
    """
    按优先级选择可用的 RAIR 特征表
    """
    for p in RAIR_FEATURES_CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def _pick_existing_index_csv():
    """
    按优先级选择可用的 RAIR 索引表
    """
    for p in RAIR_INDEX_CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def _find_first_existing_col(df: pd.DataFrame, candidates):
    """
    在 DataFrame 里按顺序查找第一个存在的列名
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


def _build_rair_feature_row(row: pd.Series, feature_csv_path: Path):
    """
    将新旧两套字段统一映射到一个兼容返回结构里：
    - 对外继续保留 baseline_pressure / min_pressure，保证旧前端不炸
    - 同时新增 baseline_signal / min_signal / value_semantics 等严谨字段
    """

    # 兼容新旧字段名
    baseline_signal = _safe_scalar(
        row.get("baseline_signal", row.get("baseline_pressure"))
    )
    min_signal = _safe_scalar(
        row.get("min_signal", row.get("min_pressure"))
    )

    relaxation_amplitude = _safe_scalar(row.get("relaxation_amplitude"))
    t_min = _safe_scalar(row.get("t_min"))
    recovery_possible = _safe_scalar(row.get("recovery_possible"))
    n_frames = _safe_scalar(row.get("n_frames"))
    dose_ml = _safe_scalar(row.get("dose_ml"))
    dose_valid = _safe_scalar(row.get("dose_valid"))
    event_id = _safe_scalar(row.get("event_id"))
    event_valid = _safe_scalar(row.get("event_valid"))
    npz_path = _safe_scalar(row.get("npz_path"))

    # join 表可能附带 cluster 信息；没有也不报错
    consensus_cluster = _safe_scalar(
        row.get("consensus_cluster", row.get("cluster"))
    )
    confidence = _safe_scalar(row.get("confidence"))
    switch_rate = _safe_scalar(row.get("switch_rate"))

    signal_type = _safe_scalar(
        row.get("signal_type", "surrogate_timeseries_from_tensor_mean")
    )
    value_unit = _safe_scalar(
        row.get("value_unit", "arbitrary_signal_unit")
    )

    return {
        "available": True,
        "data_source": feature_csv_path.name,
        "data_source_path": str(feature_csv_path),

        # ===== 旧键：保持兼容 =====
        "baseline_pressure": baseline_signal,
        "min_pressure": min_signal,

        # ===== 新键：严谨表达 =====
        "baseline_signal": baseline_signal,
        "min_signal": min_signal,
        "relaxation_amplitude": relaxation_amplitude,
        "t_min": t_min,
        "recovery_possible": recovery_possible,
        "n_frames": n_frames,
        "dose_ml": dose_ml,
        "dose_valid": dose_valid,
        "event_id": event_id,
        "event_valid": event_valid,
        "npz_path": npz_path,

        # ===== 语义说明 =====
        "signal_type": signal_type,
        "value_unit": value_unit,
        "value_semantics": "surrogate_signal_not_raw_pressure",
        "display_name_baseline": "基线信号值",
        "display_name_min": "最低信号值",
        "display_name_amplitude": "相对松弛幅度",
        "message": "RAIR 数值来自预处理张量导出的代理时序信号，并非原始 mmHg 压力。",

        # ===== 若 join 表中存在，则顺带返回 =====
        "consensus_cluster": consensus_cluster,
        "confidence": confidence,
        "switch_rate": switch_rate,
    }


def get_patient_rair_features(patient_id: str):
    feature_csv = _pick_existing_features_csv()
    if feature_csv is None:
        return {
            "available": False,
            "message": "未找到 RAIR 特征文件（rair_joined.csv / rair_feature_table.csv / rair_attn8_joined_strictdose.csv）",
            "debug": {
                "base_dir": str(BASE_DIR),
                "data_dir": str(DATA_DIR),
                "feature_candidates": [str(p) for p in RAIR_FEATURES_CSV_CANDIDATES],
            },
        }

    df = pd.read_csv(feature_csv)

    # patient_id 是必须字段
    if "patient_id" not in df.columns:
        raise ValueError(f"{feature_csv.name} 缺少字段: ['patient_id']")

    # 关键字段做宽松兼容：
    # 新版: baseline_signal/min_signal
    # 旧版: baseline_pressure/min_pressure
    baseline_col = _find_first_existing_col(df, ["baseline_signal", "baseline_pressure"])
    min_col = _find_first_existing_col(df, ["min_signal", "min_pressure"])

    required_base = [
        "patient_id",
        "dose_ml",
        "dose_valid",
        "event_id",
        "event_valid",
        "relaxation_amplitude",
        "t_min",
        "recovery_possible",
        "n_frames",
        "npz_path",
    ]
    missing = [c for c in required_base if c not in df.columns]

    if baseline_col is None:
        missing.append("baseline_signal/baseline_pressure")
    if min_col is None:
        missing.append("min_signal/min_pressure")

    if missing:
        raise ValueError(f"{feature_csv.name} 缺少字段: {missing}")

    df["patient_id"] = df["patient_id"].apply(normalize_pid)
    target_pid = normalize_pid(patient_id)

    sub = df[df["patient_id"] == target_pid].copy()
    if sub.empty:
        return {
            "available": False,
            "message": "当前患者暂无 RAIR 特征结果。",
            "data_source": feature_csv.name,
            "data_source_path": str(feature_csv),
        }

    # 排序策略保持兼容：
    # 优先按 relaxation_amplitude 最大；若有 confidence 则作为辅助排序
    sub["relaxation_amplitude_num"] = _coerce_numeric(sub["relaxation_amplitude"])

    if "confidence" in sub.columns:
        sub["confidence_num"] = _coerce_numeric(sub["confidence"]).fillna(-1)
        sub = sub.sort_values(
            ["relaxation_amplitude_num", "confidence_num"],
            ascending=[False, False],
        )
    else:
        sub = sub.sort_values("relaxation_amplitude_num", ascending=False)

    row = sub.iloc[0]
    result = _build_rair_feature_row(row, feature_csv)
    result["debug"] = {
        "target_pid": target_pid,
        "feature_csv_used": str(feature_csv),
        "feature_candidates": [str(p) for p in RAIR_FEATURES_CSV_CANDIDATES],
        "matched_rows": int(len(sub)),
    }
    return result


def _parse_any_path(path_str: str):
    """
    兼容 Windows 风格路径和 Linux 风格路径。
    返回:
      - raw_str
      - filename
      - parts
      - patient_id
    """
    s = str(path_str).strip()
    if not s:
        return "", "", [], None

    # Windows 风格优先
    if "\\" in s or (len(s) >= 2 and s[1] == ":"):
        wp = PureWindowsPath(s)
        parts = list(wp.parts)
        filename = wp.name
    else:
        p = Path(s)
        parts = list(p.parts)
        filename = p.name

    patient_id = None
    for part in parts:
        ss = str(part).strip()
        if ss.isdigit():
            patient_id = ss
            break

    return s, filename, parts, patient_id


def resolve_npz_path(npz_path_str: str):
    """
    将索引表里的 out_npz 解析到仓库中的 data/npz 下。
    支持:
    - Windows 绝对路径
    - Linux 相对/绝对路径
    - data/npz 下递归搜索
    """
    if not npz_path_str:
        return None

    raw_str, filename, parts, patient_id = _parse_any_path(npz_path_str)

    # 1. 原路径本身存在（本地环境兼容）
    direct_path = Path(raw_str)
    if direct_path.exists():
        return direct_path

    # 2. 优先按 patient_id + filename 精确拼接
    if patient_id and filename:
        candidate = RAIR_NPZ_DIR / "patients" / patient_id / filename
        if candidate.exists():
            return candidate

    # 3. data/npz 根下直接按文件名找
    if filename:
        candidate = RAIR_NPZ_DIR / filename
        if candidate.exists():
            return candidate

    # 4. 递归搜索同名文件
    matches = list(RAIR_NPZ_DIR.rglob(filename)) if filename else []
    if not matches:
        return None

    # 5. 多个同名文件时，优先选包含患者号的路径
    if patient_id:
        pid_matches = [m for m in matches if patient_id in str(m)]
        if len(pid_matches) == 1:
            return pid_matches[0]
        if len(pid_matches) > 1:
            pid_matches = sorted(pid_matches, key=lambda x: (len(x.parts), str(x)))
            return pid_matches[0]

    matches = sorted(matches, key=lambda x: (len(x.parts), str(x)))
    return matches[0]


def _extract_series_from_npz(npz_path: Path):
    if npz_path is None or not npz_path.exists():
        return None

    obj = np.load(npz_path, allow_pickle=True)

    preferred_keys = [
        "mean_pressure",
        "time_series",
        "signal",
        "series",
        "avg_signal",
        "mean_signal",
        "pressure_series",
        "ts",
        "x",
        "P_scalar",
    ]

    for k in preferred_keys:
        if k in obj.files:
            arr = np.asarray(obj[k])

            if arr.ndim == 1:
                return arr.astype(float).tolist()

            if arr.ndim == 2:
                if arr.shape[0] <= arr.shape[1]:
                    return arr.mean(axis=0).astype(float).tolist()
                return arr.mean(axis=1).astype(float).tolist()

    for k in obj.files:
        arr = np.asarray(obj[k])
        if arr.ndim == 1:
            return arr.astype(float).tolist()

    for k in obj.files:
        arr = np.asarray(obj[k])
        if arr.ndim == 2:
            if arr.shape[0] <= arr.shape[1]:
                return arr.mean(axis=0).astype(float).tolist()
            return arr.mean(axis=1).astype(float).tolist()

    return None


def get_patient_rair_time_series(patient_id: str):
    index_csv = _pick_existing_index_csv()
    if index_csv is None:
        return {
            "series": None,
            "debug": {
                "error": "No RAIR index csv found",
                "index_candidates": [str(p) for p in RAIR_INDEX_CSV_CANDIDATES],
                "rair_npz_dir": str(RAIR_NPZ_DIR),
                "base_dir": str(BASE_DIR),
                "data_dir": str(DATA_DIR),
            },
        }

    df = pd.read_csv(index_csv)
    required_cols = ["patient_id", "out_npz"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{index_csv.name} 缺少字段: {missing}")

    df["patient_id"] = df["patient_id"].apply(normalize_pid)
    target_pid = normalize_pid(patient_id)

    sub = df[df["patient_id"] == target_pid].copy()
    if sub.empty:
        return {
            "series": None,
            "debug": {
                "error": f"patient not found in {index_csv.name}",
                "target_pid": target_pid,
                "rair_index_csv": str(index_csv),
                "index_candidates": [str(p) for p in RAIR_INDEX_CSV_CANDIDATES],
                "rair_npz_dir": str(RAIR_NPZ_DIR),
            },
        }

    if "event_valid" in sub.columns:
        sub["event_valid_num"] = pd.to_numeric(sub["event_valid"], errors="coerce").fillna(0)
    else:
        sub["event_valid_num"] = 0

    if "dose_valid" in sub.columns:
        sub["dose_valid_num"] = pd.to_numeric(sub["dose_valid"], errors="coerce").fillna(0)
    else:
        sub["dose_valid_num"] = 0

    if "dose_ml" in sub.columns:
        sub["dose_ml_num"] = pd.to_numeric(sub["dose_ml"], errors="coerce").fillna(0)
    else:
        sub["dose_ml_num"] = 0

    sub = sub.sort_values(
        ["event_valid_num", "dose_valid_num", "dose_ml_num"],
        ascending=False,
    )

    row = sub.iloc[0]
    raw_out_npz = str(row["out_npz"]).strip()
    raw_str, filename, parts, parsed_pid = _parse_any_path(raw_out_npz)
    npz_path = resolve_npz_path(raw_out_npz)

    debug = {
        "target_pid": target_pid,
        "selected_out_npz": raw_out_npz,
        "parsed_filename": filename,
        "parsed_patient_id": parsed_pid,
        "resolved_npz_path": str(npz_path) if npz_path else None,
        "resolved_exists": bool(npz_path and npz_path.exists()),
        "rair_index_csv": str(index_csv),
        "index_candidates": [str(p) for p in RAIR_INDEX_CSV_CANDIDATES],
        "rair_npz_dir": str(RAIR_NPZ_DIR),
    }

    try:
        series = _extract_series_from_npz(npz_path)
        debug["series_is_none"] = series is None
        debug["series_len"] = len(series) if series else 0
        return {
            "series": series,
            "debug": debug,
        }
    except Exception as e:
        debug["error"] = str(e)
        return {
            "series": None,
            "debug": debug,
        }