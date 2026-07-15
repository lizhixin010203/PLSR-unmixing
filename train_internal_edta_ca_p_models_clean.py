

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import iqr
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "EDTA group"

FEATURE_FILE = str(DEFAULT_DATA_DIR / "EDTA-processed pointscan features.xlsx")
SPECTRA_FILE = str(DEFAULT_DATA_DIR / "EDTA-processed pointscan full spectrum.xlsx")
TARGET_FILE = str(DEFAULT_DATA_DIR / "EDTA-Ca P content.xlsx")
OUTPUT_DIR = str(SCRIPT_DIR / "results" / "plsr_internal")

SAMPLE_ID_COL = "Sample_ID"
TARGET_COLS = ["Ca", "P"]
MANUAL_PEAK_FEATURES = [
    "phosphate v2",
    "phosphate v4",
    "proline",
    "phosphate v1",
    "carbonate v1",
    "Amide III",
    "CH2",
    "Amide I",
]
SPECTRA_PREFIX = "wn_"
MAX_PLS_COMPONENTS_MANUAL = 5
MAX_PLS_COMPONENTS_FULL = 5
INNER_CV_SPLITS = 5
DEFAULT_FIPLS_INTERVALS = 20
FIPLS_MIN_RMSE_IMPROVEMENT = 1e-6
RANDOM_STATE = 42

FIRST_MODEL_NAMES = [
    "Manual-12-PLSR",
    "Manual-12-RFE-PLSR",
]
SECOND_MODEL_NAMES = [
    "Full-spectrum-PLSR",
    "Full-spectrum-FiPLS",
]

RATIO_FEATURE_SPECS = {
    "CH2/carbonate v1": ("CH2", "carbonate v1"),
    "Amide III/carbonate v1": ("Amide III", "carbonate v1"),
    "Amide I/carbonate v1": ("Amide I", "carbonate v1"),
    "phosphate v4/carbonate v1": ("phosphate v4", "carbonate v1"),
}
FEATURE_COLUMN_RENAME = {
    "Phosphate_v2": "phosphate v2",
    "Phosphate_v4": "phosphate v4",
    "Proline": "proline",
    "Phosphate_v1": "phosphate v1",
    "Carbonate_v1": "carbonate v1",
    "Amide_III": "Amide III",
    "Amide_I": "Amide I",
}
def normalize_sample_id(series: pd.Series) -> pd.Series:
    return pd.Series(series).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def read_excel_first_sheet(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(Path(path))


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """读取人工峰面积特征。"""
    df = read_excel_first_sheet(path)
    if SAMPLE_ID_COL not in df.columns:
        df.insert(0, SAMPLE_ID_COL, [f"Sample_{i + 1}" for i in range(len(df))])
    df[SAMPLE_ID_COL] = normalize_sample_id(df[SAMPLE_ID_COL])
    df = df.rename(columns=FEATURE_COLUMN_RENAME)
    return df


def load_spectrum_table(path: str | Path, spectra_prefix: str) -> pd.DataFrame:
    """读取预处理全谱并转置为样本 x 波数。"""
    raw = read_excel_first_sheet(path)
    if raw.shape[1] < 2:
        raise ValueError("全谱表至少需要 1 列波数和 1 列样本强度。")

    shift_col = raw.columns[0]
    shifts = raw[shift_col].to_numpy()
    spectrum_values = raw.drop(columns=[shift_col]).T
    spectrum_values.columns = [f"{spectra_prefix}{str(v).strip().replace('.0', '')}" for v in shifts]
    sample_ids = normalize_sample_id(list(spectrum_values.index)).to_numpy()
    spectrum_values.insert(0, SAMPLE_ID_COL, sample_ids)
    return spectrum_values.reset_index(drop=True)


def load_target_table(path: str | Path) -> pd.DataFrame:
    """读取 Ca/P 含量表。"""
    df = read_excel_first_sheet(path)
    df[SAMPLE_ID_COL] = normalize_sample_id(df[SAMPLE_ID_COL])
    return df[[SAMPLE_ID_COL] + TARGET_COLS]


def load_training_data() -> pd.DataFrame:
    """读取人工特征、Ca/P 含量和全谱数据，并按 Sample_ID 合并。"""
    feature_df = load_feature_table(FEATURE_FILE)
    target_df = load_target_table(TARGET_FILE)
    spectra_df = load_spectrum_table(SPECTRA_FILE, SPECTRA_PREFIX)

    df = feature_df.merge(target_df, on=SAMPLE_ID_COL, how="inner", validate="one_to_one")
    df = df.merge(spectra_df, on=SAMPLE_ID_COL, how="inner", validate="one_to_one")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 8 个峰面积特征基础上构建 4 个比值特征。"""
    if "carbonate v1" not in df.columns:
        raise ValueError("缺少 carbonate v1，无法计算比值特征。")
    if (df["carbonate v1"] == 0).any():
        raise ValueError("carbonate v1 存在 0，不能计算比值特征。")

    df = df.copy()
    for new_col, (num_col, denom_col) in RATIO_FEATURE_SPECS.items():
        if num_col not in df.columns:
            raise ValueError(f"缺少用于构建比值特征的列：{num_col}")
        df[new_col] = df[num_col] / df[denom_col]
    return df


def validate_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    if SAMPLE_ID_COL not in df.columns:
        df = df.copy()
        df.insert(0, SAMPLE_ID_COL, [f"Sample_{i + 1}" for i in range(len(df))])
    return df


def get_manual_feature_set(df: pd.DataFrame) -> list[str]:
    """Return the 12 manual features used by Manual-12 models."""
    base_features = list(MANUAL_PEAK_FEATURES)
    ratio_features = list(RATIO_FEATURE_SPECS.keys())
    manual_12 = base_features + ratio_features
    return manual_12


def get_full_spectrum_features(df: pd.DataFrame, spectra_prefix: str) -> list[str]:
    """识别完整预处理拉曼光谱变量。"""
    return [col for col in df.columns if str(col).startswith(spectra_prefix)]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate cross-validated regression metrics.

    Q2_CV is calculated as 1 - PRESS/TSS from cross-validated predictions.
    For LOOCV predictions this is numerically equivalent to cross-validated R2,
    but both names are exported because reviewers often expect Q2 for PLSR.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residual = y_true - y_pred
    press = float(np.sum(residual**2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    q2_cv = float(1.0 - press / tss) if tss > 0 else float("nan")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "R2_CV": float(r2),
        "Q2_CV": q2_cv,
        "RMSE_CV": float(rmse),
        "RMSECV": float(rmse),
        "MAE_CV": float(mae),
        "MAPE_CV": float(mape),
        "PRESS_CV": press,
        "TSS_CV": tss,
        "Residual_SD_CV": float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan"),
        "RPD_CV": float(np.std(y_true, ddof=1) / rmse) if rmse > 0 else float("inf"),
        "RPIQ_CV": float(iqr(y_true) / rmse) if rmse > 0 else float("inf"),
    }


def select_best_pls_components(
    X_train: np.ndarray, y_train: np.ndarray, max_components: int
) -> int:
    """Select PLS components by leakage-free inner KFold within the current training fold."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    n_samples = X_train.shape[0]
    n_splits = min(INNER_CV_SPLITS, n_samples)
    if n_splits < 2:
        raise ValueError("PLSR inner CV requires at least 2 training samples.")

    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    split_indices = list(inner_cv.split(X_train))
    min_inner_train_size = min(len(train_idx) for train_idx, _ in split_indices)
    upper = min(max_components, X_train.shape[1], min_inner_train_size - 1)
    if upper < 1:
        raise ValueError("PLSR requires at least one valid component in every inner split.")

    best_n = 1
    best_rmse = float("inf")
    for n_components in range(1, upper + 1):
        preds = np.zeros_like(y_train, dtype=float)
        for train_idx, val_idx in split_indices:
            scaler = StandardScaler()
            X_inner_train = scaler.fit_transform(X_train[train_idx])
            X_inner_val = scaler.transform(X_train[val_idx])
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_inner_train, y_train[train_idx])
            preds[val_idx] = pls.predict(X_inner_val).ravel()
        rmse = math.sqrt(mean_squared_error(y_train, preds))
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_n = n_components
    return best_n


def calculate_vip(pls_model: PLSRegression, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """统一计算 PLSR VIP 分数。"""
    del y
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    if total_s == 0:
        return np.full(p, np.nan)
    vip = np.zeros(p)
    for i in range(p):
        weight = np.array([(w[i, comp] / np.linalg.norm(w[:, comp])) ** 2 for comp in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s).item()
    return vip


def _rfe_selector(n_features_to_select: int) -> RFE:
    return RFE(
        estimator=PLSRegression(n_components=1, scale=False),
        n_features_to_select=n_features_to_select,
        step=1,
    )


def nested_loocv_plsr(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], max_components: int
) -> dict[str, Any]:
    """无 RFE 的 PLSR：StandardScaler 和主成分数选择都放在外层 LOOCV 训练折内。"""
    del feature_names
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    preds = np.zeros_like(y, dtype=float)
    selected_components = []
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        best_n = select_best_pls_components(X_train, y_train, max_components)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = PLSRegression(n_components=best_n, scale=False)
        model.fit(X_train_scaled, y_train)
        preds[test_idx] = model.predict(X_test_scaled).ravel()
        selected_components.append(best_n)

    return {
        "y_pred": preds,
        "metrics": compute_metrics(y, preds),
        "selected_components": selected_components,
    }


def _evaluate_rfe_candidate(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int,
    max_components: int,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    preds = np.zeros_like(y, dtype=float)
    best_components = []
    n_splits = min(INNER_CV_SPLITS, X.shape[0])
    if n_splits < 2:
        raise ValueError("RFE inner CV requires at least 2 training samples.")
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in inner_cv.split(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_val_scaled = scaler.transform(X[val_idx])
        selector = _rfe_selector(n_features_to_select)
        selector.fit(X_train_scaled, y[train_idx])
        X_train_sel_raw = X[train_idx][:, selector.support_]
        best_n = select_best_pls_components(X_train_sel_raw, y[train_idx], max_components)
        model = PLSRegression(n_components=best_n, scale=False)
        model.fit(selector.transform(X_train_scaled), y[train_idx])
        preds[val_idx] = model.predict(selector.transform(X_val_scaled)).ravel()
        best_components.append(best_n)
    metrics = compute_metrics(y, preds)
    return {"rmse": metrics["RMSE_CV"], "r2": metrics["R2_CV"], "best_components": best_components}


def nested_loocv_rfe_plsr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    candidate_feature_counts: list[int],
    max_components: int,
) -> dict[str, Any]:
    """RFE-PLSR：RFE、特征数选择和 PLSR 主成分数选择均嵌套在 LOOCV 训练折内。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    candidate_feature_counts = sorted(
        {int(k) for k in candidate_feature_counts if 1 <= int(k) <= X.shape[1]}
    )
    if not candidate_feature_counts:
        raise ValueError("RFE 候选特征数为空，请检查配置。")

    preds = np.zeros_like(y, dtype=float)
    outer_choices = []
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        candidate_rows = []
        for k in candidate_feature_counts:
            result = _evaluate_rfe_candidate(X_train, y_train, k, max_components)
            candidate_rows.append(
                {
                    "Candidate_Feature_Count": k,
                    "LOOCV_RMSE": result["rmse"],
                    "LOOCV_R2": result["r2"],
                    "Best_n_components": int(round(np.median(result["best_components"]))),
                }
            )
        best_row = min(candidate_rows, key=lambda row: (row["LOOCV_RMSE"], row["Candidate_Feature_Count"]))
        best_k = int(best_row["Candidate_Feature_Count"])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        selector = _rfe_selector(best_k)
        selector.fit(X_train_scaled, y_train)
        selected_names = [name for name, keep in zip(feature_names, selector.support_) if keep]
        best_n = select_best_pls_components(X_train[:, selector.support_], y_train, max_components)
        model = PLSRegression(n_components=best_n, scale=False)
        model.fit(selector.transform(X_train_scaled), y_train)
        preds[test_idx] = model.predict(selector.transform(X_test_scaled)).ravel()
        outer_choices.append(
            {
                "Candidate_Feature_Count": best_k,
                "Best_n_components": best_n,
                "Selected_Features": "; ".join(selected_names),
                "Inner_Candidate_Table": candidate_rows,
            }
        )

    return {"y_pred": preds, "metrics": compute_metrics(y, preds), "outer_choices": outer_choices}


def make_intervals(n_features: int, n_intervals: int) -> list[np.ndarray]:
    """Split ordered full-spectrum columns into fixed contiguous intervals without using y."""
    if n_features < 1 or n_intervals < 1:
        raise ValueError("n_features and n_intervals must be positive.")
    return [part for part in np.array_split(np.arange(n_features), n_intervals) if part.size > 0]


def evaluate_interval_subset_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_feature_idx: np.ndarray,
    max_components: int,
) -> dict[str, Any]:
    """Evaluate one interval subset by leakage-free inner KFold within the outer training fold."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    selected_feature_idx = np.asarray(selected_feature_idx, dtype=int)
    if selected_feature_idx.size == 0:
        raise ValueError("iPLS selected_feature_idx cannot be empty.")

    X_subset = X_train[:, selected_feature_idx]
    n_splits = min(INNER_CV_SPLITS, len(y_train))
    if n_splits < 2:
        raise ValueError("iPLS inner CV requires at least 2 samples.")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_subset))
    min_inner_train_size = min(len(train_idx) for train_idx, _ in splits)
    upper = min(max_components, X_subset.shape[1], min_inner_train_size - 1)
    if upper < 1:
        raise ValueError("No valid PLS component count for this iPLS interval subset.")

    predictions = np.zeros((len(y_train), upper), dtype=float)
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_subset[train_idx])
        X_val = scaler.transform(X_subset[val_idx])
        for n_components in range(1, upper + 1):
            model = PLSRegression(n_components=n_components, scale=False)
            model.fit(X_fit, y_train[train_idx])
            predictions[val_idx, n_components - 1] = model.predict(X_val).ravel()

    best_n = 1
    best_rmse = float("inf")
    for n_components in range(1, upper + 1):
        rmse = math.sqrt(mean_squared_error(y_train, predictions[:, n_components - 1]))
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_n = n_components
    best_pred = predictions[:, best_n - 1]
    return {
        "RMSE_CV": float(best_rmse),
        "R2_CV": float(r2_score(y_train, best_pred)),
        "Best_n_components": int(best_n),
        "Selected_Feature_Count": int(selected_feature_idx.size),
    }


def select_fipls_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    intervals: list[np.ndarray],
    max_components: int,
    min_rmse_improvement: float = FIPLS_MIN_RMSE_IMPROVEMENT,
) -> dict[str, Any]:
    """Greedy forward iPLS using only the current outer training fold."""
    if not intervals:
        raise ValueError("FiPLS requires at least one spectral interval.")

    selected_ids: list[int] = []
    remaining_ids = list(range(1, len(intervals) + 1))
    candidate_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    current_result: dict[str, Any] | None = None

    for step in range(1, len(intervals) + 1):
        step_candidates: list[dict[str, Any]] = []
        for candidate_id in remaining_ids:
            trial_ids = selected_ids + [candidate_id]
            selected_idx = np.unique(
                np.concatenate([intervals[interval_id - 1] for interval_id in trial_ids])
            )
            result = evaluate_interval_subset_cv(
                X_train, y_train, selected_idx, max_components
            )
            row = {
                "Forward_Step": step,
                "Candidate_Added_Interval_ID": candidate_id,
                "Trial_Interval_IDs": "; ".join(map(str, trial_ids)),
                "Trial_Interval_Count": len(trial_ids),
                "Selected_Feature_Count": int(selected_idx.size),
                "RMSE_CV": result["RMSE_CV"],
                "R2_CV": result["R2_CV"],
                "Best_n_components": result["Best_n_components"],
                "Accepted_At_Step": False,
            }
            candidate_rows.append(row)
            step_candidates.append(
                {
                    **row,
                    "selected_idx": selected_idx,
                    "candidate_row_index": len(candidate_rows) - 1,
                }
            )

        if not step_candidates:
            break
        minimum_rmse = min(candidate["RMSE_CV"] for candidate in step_candidates)
        near_best = [
            candidate
            for candidate in step_candidates
            if candidate["RMSE_CV"] <= minimum_rmse + 1e-6
        ]
        best_candidate = min(
            near_best,
            key=lambda candidate: candidate["Candidate_Added_Interval_ID"],
        )

        improved = (
            current_result is None
            or best_candidate["RMSE_CV"]
            < current_result["RMSE_CV"] - min_rmse_improvement
        )
        if not improved:
            break

        candidate_rows[best_candidate["candidate_row_index"]]["Accepted_At_Step"] = True
        selected_ids.append(int(best_candidate["Candidate_Added_Interval_ID"]))
        remaining_ids.remove(selected_ids[-1])
        current_result = best_candidate
        path_rows.append(
            {
                "Forward_Step": step,
                "Added_Interval_ID": selected_ids[-1],
                "Selected_Interval_IDs": "; ".join(map(str, selected_ids)),
                "Selected_Interval_Count": len(selected_ids),
                "Selected_Feature_Count": int(len(best_candidate["selected_idx"])),
                "RMSE_CV": float(best_candidate["RMSE_CV"]),
                "R2_CV": float(best_candidate["R2_CV"]),
                "Best_n_components": int(best_candidate["Best_n_components"]),
            }
        )

    if current_result is None:
        raise RuntimeError("FiPLS did not produce a valid interval subset.")
    return {
        "selected_interval_ids": selected_ids,
        "selected_feature_idx": np.asarray(current_result["selected_idx"], dtype=int),
        "candidate_evaluation_table": pd.DataFrame(candidate_rows),
        "forward_path_table": pd.DataFrame(path_rows),
        "best_n_components": int(current_result["Best_n_components"]),
        "best_rmse_cv": float(current_result["RMSE_CV"]),
    }


def nested_loocv_fipls(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_components: int,
    progress_label: str,
    n_intervals: int = DEFAULT_FIPLS_INTERVALS,
) -> dict[str, Any]:
    """
    Strict nested-LOOCV forward interval PLS.

    Full-spectrum-FiPLS is a full-spectrum data-driven interval-selection baseline.
    At every forward step, all remaining intervals are conditionally tested using
    only the outer training fold; the held-out sample is used only for prediction.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    intervals = make_intervals(X.shape[1], n_intervals)
    predictions = np.zeros_like(y, dtype=float)
    fold_rows = []
    splits = list(LeaveOneOut().split(X))
    iterator = tqdm(splits, total=len(splits), desc=f"{progress_label} outer LOOCV", unit="fold")

    for fold_id, (train_idx, test_idx) in enumerate(iterator, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        selection = select_fipls_intervals(X_train, y_train, intervals, max_components)
        selected_idx = selection["selected_feature_idx"]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[:, selected_idx])
        X_test_scaled = scaler.transform(X_test[:, selected_idx])
        model = PLSRegression(n_components=selection["best_n_components"], scale=False)
        model.fit(X_train_scaled, y_train)
        predictions[test_idx] = model.predict(X_test_scaled).ravel()
        selected_names = [feature_names[i] for i in selected_idx]
        fold_rows.append(
            {
                "Fold": fold_id,
                "Test_Sample_Index": int(test_idx[0]),
                "N_FIPLS_Intervals": int(n_intervals),
                "Selected_Interval_Count": len(selection["selected_interval_ids"]),
                "Selected_Interval_IDs": "; ".join(map(str, selection["selected_interval_ids"])),
                "Selected_Feature_Count": int(selected_idx.size),
                "Best_n_components": int(selection["best_n_components"]),
                "Selected_Features": "; ".join(selected_names),
                "Inner_RMSE_CV": float(selection["best_rmse_cv"]),
            }
        )
        iterator.set_postfix(intervals=len(selection["selected_interval_ids"]), refresh=False)

    return {
        "y_pred": predictions,
        "metrics": compute_metrics(y, predictions),
        "fipls_selection_summary": pd.DataFrame(fold_rows),
    }


def fit_final_plsr_model(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], max_components: int
) -> dict[str, Any]:
    """训练最终 PLSR 模型。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    best_n = select_best_pls_components(X, y, max_components)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = PLSRegression(n_components=best_n, scale=False)
    model.fit(X_scaled, y)
    vip = calculate_vip(model, X_scaled, y)
    return {
        "scaler": scaler,
        "selector": None,
        "model": model,
        "selected_feature_names": list(feature_names),
        "best_n_components": best_n,
        "vip": vip,
        "X_final_scaled_selected": X_scaled,
    }


def fit_final_rfe_plsr_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    candidate_feature_counts: list[int],
    max_components: int,
) -> dict[str, Any]:
    """用全部 50 个内部样本训练带 RFE 的最终 PLSR 模型。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    candidate_feature_counts = sorted(
        {int(k) for k in candidate_feature_counts if 1 <= int(k) <= X.shape[1]}
    )
    rows = []
    for k in candidate_feature_counts:
        result = _evaluate_rfe_candidate(X, y, k, max_components)
        rows.append(
            {
                "Candidate_Feature_Count": k,
                "LOOCV_RMSE": result["rmse"],
                "LOOCV_R2": result["r2"],
                "Best_n_components": int(round(np.median(result["best_components"]))),
            }
        )
    best_row = min(rows, key=lambda row: (row["LOOCV_RMSE"], row["Candidate_Feature_Count"]))
    best_k = int(best_row["Candidate_Feature_Count"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = _rfe_selector(best_k)
    selector.fit(X_scaled, y)
    selected_feature_names = [name for name, keep in zip(feature_names, selector.support_) if keep]
    best_n = select_best_pls_components(X[:, selector.support_], y, max_components)
    X_selected = selector.transform(X_scaled)
    model = PLSRegression(n_components=best_n, scale=False)
    model.fit(X_selected, y)
    vip = calculate_vip(model, X_selected, y)

    for row in rows:
        row["Selected_Features"] = "; ".join(selected_feature_names) if row["Candidate_Feature_Count"] == best_k else ""

    return {
        "scaler": scaler,
        "selector": selector,
        "model": model,
        "selected_feature_names": selected_feature_names,
        "best_n_components": best_n,
        "rfe_selection_summary": pd.DataFrame(rows),
        "vip": vip,
        "X_final_scaled_selected": X_selected,
    }


def fit_final_fipls_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_components: int,
    n_intervals: int = DEFAULT_FIPLS_INTERVALS,
) -> dict[str, Any]:
    """Train final forward-iPLS on all 50 internal samples for later validation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    intervals = make_intervals(X.shape[1], n_intervals)
    selection = select_fipls_intervals(X, y, intervals, max_components)
    selected_idx = selection["selected_feature_idx"]
    selected_names = [feature_names[i] for i in selected_idx]
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X[:, selected_idx])
    best_n = selection["best_n_components"]
    model = PLSRegression(n_components=best_n, scale=False)
    model.fit(X_selected, y)
    vip = calculate_vip(model, X_selected, y)

    return {
        "scaler": scaler,
        "selector": selected_idx,
        "intervals": intervals,
        "selected_interval_ids": selection["selected_interval_ids"],
        "model": model,
        "selected_feature_names": selected_names,
        "best_n_components": best_n,
        "vip": vip,
        "X_final_scaled_selected": X_selected,
        "fipls_candidate_evaluation_table": selection["candidate_evaluation_table"],
        "fipls_forward_path_table": selection["forward_path_table"],
        "final_selected_feature_count": len(selected_names),
    }


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    target: str,
    model_name: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(y_true, y_pred, s=42, edgecolor="black", linewidth=0.5, alpha=0.85)
    lim_min = min(np.min(y_true), np.min(y_pred))
    lim_max = max(np.max(y_true), np.max(y_pred))
    pad = (lim_max - lim_min) * 0.06 if lim_max > lim_min else 1.0
    plt.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "r--", lw=1.2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted_LOOCV")
    plt.title(f"{target} - {model_name}")
    text = (
        f"R2={metrics['R2_CV']:.3f}\n"
        f"RMSE={metrics['RMSE_CV']:.3f}\n"
        f"MAE={metrics['MAE_CV']:.3f}\n"
        f"RPD={metrics['RPD_CV']:.3f}"
    )
    plt.text(0.04, 0.96, text, transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def compute_prediction_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return generic regression diagnostics for calibration or external prediction."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residual = y_true - y_pred
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "Residual_SD": float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan"),
        "SSE": float(np.sum(residual**2)),
        "TSS": float(np.sum((y_true - np.mean(y_true)) ** 2)),
    }


def save_pls_weights_loadings_scores(
    model_dir: Path,
    file_prefix: str,
    final_artifact: dict[str, Any],
    selected_features: list[str],
    sample_ids: pd.Series,
) -> None:
    """Export PLS model matrices required for detailed model-weight diagnostics."""
    model = final_artifact["model"]
    component_names = [f"LV{i + 1}" for i in range(int(model.n_components))]
    workbook_path = model_dir / f"{file_prefix}_pls_weights_loadings_scores.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        if hasattr(model, "x_weights_"):
            x_weights = pd.DataFrame(model.x_weights_, columns=component_names)
            x_weights.insert(0, "Feature", selected_features)
            x_weights.to_excel(writer, index=False, sheet_name="X_weights")
        if hasattr(model, "x_loadings_"):
            x_loadings = pd.DataFrame(model.x_loadings_, columns=component_names)
            x_loadings.insert(0, "Feature", selected_features)
            x_loadings.to_excel(writer, index=False, sheet_name="X_loadings")
        if hasattr(model, "y_loadings_"):
            y_loadings = np.asarray(model.y_loadings_)
            if y_loadings.ndim == 2 and y_loadings.shape[0] == 1:
                y_loadings_df = pd.DataFrame(y_loadings, columns=component_names)
                y_loadings_df.insert(0, "Target", "Y")
            else:
                y_loadings_df = pd.DataFrame(y_loadings)
            y_loadings_df.to_excel(writer, index=False, sheet_name="Y_loadings")
        if hasattr(model, "x_scores_"):
            x_scores = pd.DataFrame(model.x_scores_, columns=component_names)
            x_scores.insert(0, "Sample_ID", pd.Series(sample_ids).astype(str).to_numpy())
            x_scores.to_excel(writer, index=False, sheet_name="X_scores")
        if hasattr(model, "y_scores_"):
            y_scores = np.asarray(model.y_scores_)
            y_scores_df = pd.DataFrame(y_scores, columns=component_names[: y_scores.shape[1]])
            y_scores_df.insert(0, "Sample_ID", pd.Series(sample_ids).astype(str).to_numpy())
            y_scores_df.to_excel(writer, index=False, sheet_name="Y_scores")


def save_cv_model_selection_diagnostics(
    model_dir: Path,
    file_prefix: str,
    cv_result: dict[str, Any] | None,
) -> None:
    """Export fold-wise latent-variable and feature-selection diagnostics."""
    if not cv_result:
        return
    if "selected_components" in cv_result:
        pd.DataFrame(
            {
                "Fold": np.arange(1, len(cv_result["selected_components"]) + 1),
                "Best_n_components": cv_result["selected_components"],
            }
        ).to_excel(model_dir / f"{file_prefix}_outer_cv_components.xlsx", index=False)
    if "outer_choices" in cv_result:
        outer_rows = []
        inner_rows = []
        for fold_id, choice in enumerate(cv_result["outer_choices"], start=1):
            outer_rows.append(
                {
                    "Fold": fold_id,
                    "Candidate_Feature_Count": choice.get("Candidate_Feature_Count"),
                    "Best_n_components": choice.get("Best_n_components"),
                    "Selected_Features": choice.get("Selected_Features"),
                }
            )
            for row in choice.get("Inner_Candidate_Table", []):
                inner_rows.append({"Fold": fold_id, **row})
        with pd.ExcelWriter(model_dir / f"{file_prefix}_outer_cv_rfe_diagnostics.xlsx", engine="openpyxl") as writer:
            pd.DataFrame(outer_rows).to_excel(writer, index=False, sheet_name="Outer_Selected_Model")
            pd.DataFrame(inner_rows).to_excel(writer, index=False, sheet_name="Inner_Candidate_Table")
    if "fipls_selection_summary" in cv_result:
        cv_result["fipls_selection_summary"].to_excel(
            model_dir / f"{file_prefix}_outer_cv_fipls_selection.xlsx", index=False
        )

def save_model_outputs(
    target_dir: Path,
    model_name: str,
    target: str,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    final_artifact: dict[str, Any],
    input_type: str,
    feature_selection_method: str,
    is_plsr: bool,
    summary_extra: dict[str, Any] | None = None,
    cv_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_dir = target_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{target}_{model_name}"

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    final_artifact["y_train_mean"] = float(np.mean(y_true))
    final_artifact["y_train_std"] = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")

    prediction_df = pd.DataFrame(
        {
            "Sample_ID": sample_ids,
            "Target": target,
            "Model_Name": model_name,
            "Prediction_Type": "LOOCV",
            "Actual": y_true,
            "Predicted_LOOCV": y_pred,
            "Residual": y_true - y_pred,
        }
    )
    prediction_df.to_excel(model_dir / f"{file_prefix}_prediction_results.xlsx", index=False)

    calibration_pred = np.asarray(final_artifact["model"].predict(final_artifact["X_final_scaled_selected"])).ravel()
    calibration_df = pd.DataFrame(
        {
            "Sample_ID": sample_ids,
            "Target": target,
            "Model_Name": model_name,
            "Prediction_Type": "Calibration",
            "Actual": y_true,
            "Predicted_Calibration": calibration_pred,
            "Residual": y_true - calibration_pred,
        }
    )
    calibration_df.to_excel(model_dir / f"{file_prefix}_calibration_results.xlsx", index=False)

    cal_metrics = compute_prediction_diagnostics(y_true, calibration_pred)
    diagnostic_df = pd.DataFrame(
        [
            {
                "Target": target,
                "Model_Name": model_name,
                "Data_Set": "Calibration_internal_50",
                "R2": cal_metrics["R2"],
                "Q2": np.nan,
                "RMSE": cal_metrics["RMSE"],
                "RMSEC": cal_metrics["RMSE"],
                "RMSECV": np.nan,
                "RMSEP": np.nan,
                "MAE": cal_metrics["MAE"],
                "MAPE": cal_metrics["MAPE"],
                "Residual_SD": cal_metrics["Residual_SD"],
                "Number_of_Latent_Variables": final_artifact.get("best_n_components", np.nan),
                "Selected_Feature_Count": len(final_artifact.get("selected_feature_names", [])),
            },
            {
                "Target": target,
                "Model_Name": model_name,
                "Data_Set": "Cross_validation_LOOCV_internal_50",
                "R2": metrics.get("R2_CV", np.nan),
                "Q2": metrics.get("Q2_CV", metrics.get("R2_CV", np.nan)),
                "RMSE": metrics.get("RMSE_CV", np.nan),
                "RMSEC": np.nan,
                "RMSECV": metrics.get("RMSE_CV", np.nan),
                "RMSEP": np.nan,
                "MAE": metrics.get("MAE_CV", np.nan),
                "MAPE": metrics.get("MAPE_CV", np.nan),
                "Residual_SD": metrics.get("Residual_SD_CV", np.nan),
                "Number_of_Latent_Variables": final_artifact.get("best_n_components", np.nan),
                "Selected_Feature_Count": len(final_artifact.get("selected_feature_names", [])),
            },
        ]
    )
    diagnostic_df.to_excel(model_dir / f"{file_prefix}_model_diagnostics.xlsx", index=False)

    plot_actual_vs_predicted(
        y_true,
        y_pred,
        metrics,
        target,
        model_name,
        model_dir / f"{file_prefix}_actual_vs_predicted.png",
    )

    selected_features = final_artifact.get("selected_feature_names", [])
    if is_plsr:
        pd.DataFrame({"Feature": selected_features}).to_excel(
            model_dir / f"{file_prefix}_final_selected_features.xlsx", index=False
        )
        coef = np.asarray(final_artifact["model"].coef_).ravel()
        pd.DataFrame({"Feature": selected_features, "Coefficient": coef}).to_excel(
            model_dir / f"{file_prefix}_final_pls_coefficients.xlsx", index=False
        )
        vip = np.asarray(final_artifact["vip"]).ravel()
        vip_features = final_artifact.get("vip_feature_names", selected_features)
        vip_selected_flags = final_artifact.get(
            "vip_selected_flags", [True for _ in range(len(vip_features))]
        )
        pd.DataFrame(
            {
                "Feature": vip_features,
                "VIP": vip,
                "Selected_in_Final_Model": vip_selected_flags,
                "Target": target,
                "Model_Name": model_name,
            }
        ).to_excel(model_dir / f"{file_prefix}_vip_scores.xlsx", index=False)
        save_pls_weights_loadings_scores(model_dir, file_prefix, final_artifact, list(selected_features), sample_ids)
        save_cv_model_selection_diagnostics(model_dir, file_prefix, cv_result)
        if final_artifact.get("rfe_selection_summary") is not None:
            final_artifact["rfe_selection_summary"].to_excel(
                model_dir / f"{file_prefix}_rfe_selection_summary.xlsx", index=False
            )
        if final_artifact.get("fipls_selection_summary") is not None:
            final_artifact["fipls_selection_summary"].to_excel(
                model_dir / f"{file_prefix}_fipls_selection_summary.xlsx", index=False
            )
        if final_artifact.get("fipls_candidate_evaluation_table") is not None:
            final_artifact["fipls_candidate_evaluation_table"].to_excel(
                model_dir / f"{file_prefix}_fipls_candidate_evaluation_table.xlsx", index=False
            )
        if final_artifact.get("fipls_forward_path_table") is not None:
            final_artifact["fipls_forward_path_table"].to_excel(
                model_dir / f"{file_prefix}_fipls_forward_path_table.xlsx", index=False
            )

    model_file = model_dir / f"{target}_{model_name}_final_model.pkl"
    joblib.dump(final_artifact, model_file)

    row = {
        "Target": target,
        "Model_Name": model_name,
        "Input_Type": input_type,
        "Feature_Selection_Method": feature_selection_method,
        **metrics,
        "R2_Cal": cal_metrics["R2"],
        "RMSEC": cal_metrics["RMSE"],
        "MAE_Cal": cal_metrics["MAE"],
        "MAPE_Cal": cal_metrics["MAPE"],
        "Selected_Feature_Count": len(selected_features) if selected_features else np.nan,
        "Best_n_components": final_artifact.get("best_n_components", np.nan),
        "Model_File_Path": str(model_file),
        "Result_Folder": str(model_dir),
    }
    if summary_extra:
        row.update(summary_extra)
    row["Final_Selected_Feature_Count"] = final_artifact.get(
        "final_selected_feature_count",
        len(selected_features) if selected_features else np.nan,
    )
    return row


def _build_model_plan(
    manual_12: list[str],
    full_features: list[str],
    model_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    model_plan = [
        {
            "name": "Manual-12-PLSR",
            "features": manual_12,
            "input_type": "8 manual peak-area features + 4 ratio features",
            "feature_selection": "None",
            "kind": "plsr",
            "max_components": MAX_PLS_COMPONENTS_MANUAL,
        },
        {
            "name": "Manual-12-RFE-PLSR",
            "features": manual_12,
            "input_type": "12 manual features",
            "feature_selection": "RFE inside LOOCV",
            "kind": "rfe_plsr",
            "candidate_counts": [k for k in [3, 4, 5, 6, 8, 10, 12] if k <= len(manual_12)],
            "max_components": MAX_PLS_COMPONENTS_MANUAL,
        },
        {
            "name": "Full-spectrum-PLSR",
            "features": full_features,
            "input_type": "full preprocessed Raman spectrum",
            "feature_selection": "None",
            "kind": "plsr",
            "max_components": MAX_PLS_COMPONENTS_FULL,
        },
        {
            "name": "Full-spectrum-FiPLS",
            "features": full_features,
            "input_type": "full preprocessed Raman spectrum",
            "feature_selection": "forward interval PLS inside LOOCV",
            "kind": "fipls",
            "max_components": MAX_PLS_COMPONENTS_FULL,
        },
    ]
    if model_names is None:
        return model_plan
    allowed = set(model_names)
    selected_plan = [spec for spec in model_plan if spec["name"] in allowed]
    missing = allowed - {spec["name"] for spec in model_plan}
    if missing:
        raise ValueError("model_names 中存在未知模型：" + ", ".join(sorted(missing)))
    if not selected_plan:
        raise ValueError("model_names 没有选中任何模型。")
    return selected_plan


def run_all_models_for_target(
    target: str,
    df: pd.DataFrame,
    output_dir: Path,
    manual_12: list[str],
    full_features: list[str],
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """分别为 Ca 或 P 运行全部模型。"""
    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = df[SAMPLE_ID_COL]
    y = df[target].to_numpy(dtype=float)
    rows = []

    for spec in _build_model_plan(manual_12, full_features, model_names):
        model_name = spec["name"]
        feature_names = spec["features"]
        X = df[feature_names].to_numpy(dtype=float)
        print(f"\n[{target}] 正在运行 {model_name} ...")

        if spec["kind"] == "plsr":
            cv_result = nested_loocv_plsr(X, y, feature_names, spec["max_components"])
            final_artifact = fit_final_plsr_model(X, y, feature_names, spec["max_components"])
            is_plsr = True
        elif spec["kind"] == "rfe_plsr":
            cv_result = nested_loocv_rfe_plsr(
                X,
                y,
                feature_names,
                spec["candidate_counts"],
                spec["max_components"],
            )
            final_artifact = fit_final_rfe_plsr_model(
                X,
                y,
                feature_names,
                spec["candidate_counts"],
                spec["max_components"],
            )
            is_plsr = True
        elif spec["kind"] == "fipls":
            cv_result = nested_loocv_fipls(
                X,
                y,
                feature_names,
                spec["max_components"],
                progress_label=f"{target} FiPLS",
                n_intervals=DEFAULT_FIPLS_INTERVALS,
            )
            final_artifact = fit_final_fipls_model(
                X,
                y,
                feature_names,
                spec["max_components"],
                n_intervals=DEFAULT_FIPLS_INTERVALS,
            )
            final_artifact["fipls_selection_summary"] = cv_result["fipls_selection_summary"]
            final_artifact["final_selected_feature_count"] = len(final_artifact["selected_feature_names"])
            is_plsr = True
        else:
            raise ValueError(f"未知模型类型：{spec['kind']}")

        if spec["kind"] == "rfe_plsr":
            fold_counts = [row["Candidate_Feature_Count"] for row in cv_result["outer_choices"]]
        elif spec["kind"] == "fipls":
            fold_counts = cv_result["fipls_selection_summary"]["Selected_Feature_Count"].tolist()
        else:
            fold_counts = [len(feature_names)]

        variable_selection_type = {
            "Manual-12-PLSR": "None",
            "Manual-12-RFE-PLSR": "RFE inside LOOCV",
            "Full-spectrum-PLSR": "None",
            "Full-spectrum-FiPLS": "forward interval PLS inside LOOCV",
        }.get(model_name, spec["feature_selection"])
        summary_extra = {
            "Is_Full_Spectrum_Input": model_name.startswith("Full-spectrum"),
            "Is_Data_Driven_Baseline": model_name.startswith("Full-spectrum"),
            "Variable_Selection_Type": variable_selection_type,
            "Median_Selected_Feature_Count": float(np.median(fold_counts)) if fold_counts else np.nan,
        }

        row = save_model_outputs(
            target_dir=target_dir,
            model_name=model_name,
            target=target,
            sample_ids=sample_ids,
            y_true=y,
            y_pred=cv_result["y_pred"],
            metrics=cv_result["metrics"],
            final_artifact=final_artifact,
            input_type=spec["input_type"],
            feature_selection_method=spec["feature_selection"],
            is_plsr=is_plsr,
            summary_extra=summary_extra,
            cv_result=cv_result,
        )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("RMSE_CV", ascending=True)
    summary.to_excel(target_dir / "model_performance_summary.xlsx", index=False)
    return summary


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data()
    df = validate_input_columns(df)
    df = add_ratio_features(df)
    manual_12 = get_manual_feature_set(df)
    full_features = get_full_spectrum_features(df, SPECTRA_PREFIX)


    summary_parts = {target: [] for target in TARGET_COLS}
    for group_name, model_names in [
        ("manual PLSR models", FIRST_MODEL_NAMES),
        ("full-spectrum PLSR/FiPLS models", SECOND_MODEL_NAMES),
    ]:
        if not model_names:
            continue
        print(f"\n================ {group_name} ================")
        for target in TARGET_COLS:
            summary = run_all_models_for_target(
                target=target,
                df=df,
                output_dir=output_dir,
                manual_12=manual_12,
                full_features=full_features,
                model_names=model_names,
            )
            summary_parts[target].append(summary)
            combined_summary = pd.concat(summary_parts[target], ignore_index=True).sort_values(
                "RMSE_CV", ascending=True
            )
            combined_summary.to_excel(
                output_dir / target / "model_performance_summary.xlsx", index=False
            )

    all_summaries = {
        target: pd.concat(parts, ignore_index=True).sort_values("RMSE_CV", ascending=True)
        for target, parts in summary_parts.items()
    }

    print("\n================ 运行结束总结 ================")
    for target, summary in all_summaries.items():
        print(f"\n{target} 模型性能排序（按 RMSE_CV 从小到大）：")
        print(summary[["Model_Name", "RMSE_CV", "R2_CV", "MAE_CV", "RPD_CV"]].to_string(index=False))
        print(f"{target} 结果文件夹：{output_dir / target}")
        print("最终模型 .pkl 文件：")
        for path in summary["Model_File_Path"]:
            print(f"  {path}")


if __name__ == "__main__":
    main()

