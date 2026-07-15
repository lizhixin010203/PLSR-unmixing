import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment, nnls
from scipy.signal import find_peaks
from sklearn.decomposition import PCA



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "EDTA group")

@dataclass
class MCRParams:
    data_path: str = os.path.join(DEFAULT_DATA_DIR, "m1_mcr_ready.npy")
    wn_path: str = os.path.join(DEFAULT_DATA_DIR, "m1_wn.npy")
    valid_mask_path: str = os.path.join(DEFAULT_DATA_DIR, "m1_valid_mask.npy")
    save_dir: str = os.path.join(SCRIPT_DIR, "results", "m1")
    sample_name: str = "m1"

    component_numbers: Tuple[int, ...] = (2, 3, 4, 5)
    svd_components_to_report: int = 10
    max_iter: int = 100
    tol: float = 3e-4
    random_state: int = 42
    n_random_starts: int = 5
    sum_to_one_maps: bool = False

    flip_horizontal_y_for_display: bool = True
    map_pixel_size_um: float = 1.0
    scalebar_um: float = 10.0
    contour_levels: int = 96
    map_display_smoothing_sigma_px: float = 0.5
    map_contrast_low_percentile: float = 0.0
    map_contrast_high_percentile: float = 99.5
    component_peak_max_labels: int = 6
    component_peak_min_height_fraction: float = 0.06
    component_peak_min_prominence_fraction: float = 0.035
    component_peak_min_distance_points: int = 6
    component_map_colors: Tuple[str, ...] = (
        "#7BC7E8",
        "#F15A40",
        "#18C6A7",
        "#6E80C4",
        "#F39B7F",
        "#9AA7D7",
    )
    reference_spectra_path: str = os.path.join(DEFAULT_DATA_DIR, "HA_collagen_references.xlsx")
    reference_sheet_name: str = "reference"
    reference_similarity_windows: Tuple[Tuple[str, float, float], ...] = (("full_400_1800", 400.0, 1800.0),)
def load_preprocessed_data(params: MCRParams):
    cube = np.load(params.data_path)
    wn = np.load(params.wn_path)
    valid_mask = np.load(params.valid_mask_path).astype(bool)

    if cube.ndim != 3:
        raise ValueError("Preprocessed data must have shape nx x ny x n_bands.")
    if valid_mask.shape != cube.shape[:2]:
        raise ValueError("valid_mask shape does not match the mapping dimensions.")

    D = cube[valid_mask, :].copy()
    finite_rows = np.all(np.isfinite(D), axis=1)
    if not np.all(finite_rows):
        coords = np.argwhere(valid_mask)
        clean_mask = np.zeros_like(valid_mask, dtype=bool)
        clean_coords = coords[finite_rows]
        clean_mask[clean_coords[:, 0], clean_coords[:, 1]] = True
        valid_mask = clean_mask
        D = cube[valid_mask, :].copy()

    return wn, valid_mask, D


def load_reference_spectra(params: MCRParams, target_wn: np.ndarray) -> Dict[str, np.ndarray]:
    """Load HA/collagen reference spectra and interpolate them to the MCR axis."""
    if not params.reference_spectra_path:
        return {}
    if not os.path.exists(params.reference_spectra_path):
        raise FileNotFoundError(f"Reference spectra file not found: {params.reference_spectra_path}")

    reference_df = pd.read_excel(
        params.reference_spectra_path,
        sheet_name=params.reference_sheet_name,
        header=1,
    )
    reference_df = reference_df.apply(pd.to_numeric, errors="coerce")
    if reference_df.shape[1] < 3:
        raise ValueError(
            "Reference spectra sheet must contain Raman shift, HA intensity, and collagen intensity columns."
        )

    reference_wn = reference_df.iloc[:, 0].to_numpy(dtype=float)
    finite_axis = np.isfinite(reference_wn)
    if np.count_nonzero(finite_axis) < 2:
        raise ValueError("Reference Raman-shift axis contains too few finite values.")

    order = np.argsort(reference_wn[finite_axis])
    reference_wn = reference_wn[finite_axis][order]
    reference_spectra = {}
    reference_names = ["HA_reference", "collagen_reference"]

    for column_index, reference_name in zip([1, 2], reference_names):
        reference_intensity = reference_df.iloc[:, column_index].to_numpy(dtype=float)
        reference_intensity = reference_intensity[finite_axis][order]
        finite_values = np.isfinite(reference_intensity)
        if np.count_nonzero(finite_values) < 2:
            continue
        aligned = np.interp(
            target_wn,
            reference_wn[finite_values],
            reference_intensity[finite_values],
            left=np.nan,
            right=np.nan,
        )
        reference_spectra[reference_name] = aligned

    if not reference_spectra:
        raise ValueError("No usable HA/collagen reference spectra were found in the reference workbook.")
    return reference_spectra


def safe_cosine(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return np.nan
    a_masked = a[mask]
    b_masked = b[mask]
    denominator = np.linalg.norm(a_masked) * np.linalg.norm(b_masked)
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float(np.dot(a_masked, b_masked) / denominator)


def safe_pearson(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return np.nan
    a_masked = a[mask]
    b_masked = b[mask]
    if np.std(a_masked) <= np.finfo(float).eps or np.std(b_masked) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(a_masked, b_masked)[0, 1])


def normalized_euclidean_distance(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return np.nan
    a_masked = a[mask]
    b_masked = b[mask]
    a_norm = np.linalg.norm(a_masked)
    b_norm = np.linalg.norm(b_masked)
    if a_norm <= np.finfo(float).eps or b_norm <= np.finfo(float).eps:
        return np.nan
    return float(np.linalg.norm(a_masked / a_norm - b_masked / b_norm))


def reference_similarity_table(
    S: np.ndarray,
    wn: np.ndarray,
    reference_spectra: Dict[str, np.ndarray],
    params: MCRParams,
):
    records = []
    if not reference_spectra:
        return pd.DataFrame()

    for component_index, component_spectrum in enumerate(S, start=1):
        for reference_name, reference_spectrum in reference_spectra.items():
            for window_name, low, high in params.reference_similarity_windows:
                window_mask = (wn >= low) & (wn <= high)
                if np.count_nonzero(window_mask) < 3:
                    continue
                component_window = component_spectrum[window_mask]
                reference_window = reference_spectrum[window_mask]
                cosine = safe_cosine(component_window, reference_window)
                positive_cosine = safe_cosine(
                    np.maximum(component_window, 0.0),
                    np.maximum(reference_window, 0.0),
                )
                pearson = safe_pearson(component_window, reference_window)
                angle = (
                    float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                    if np.isfinite(cosine)
                    else np.nan
                )
                records.append(
                    {
                        "component": component_index,
                        "component_label": f"MC{component_index}",
                        "reference": reference_name,
                        "window": window_name,
                        "window_low_cm-1": low,
                        "window_high_cm-1": high,
                        "n_bands": int(np.count_nonzero(window_mask)),
                        "cosine_similarity": cosine,
                        "positive_cosine_similarity": positive_cosine,
                        "spectral_angle_degrees": angle,
                        "pearson_correlation": pearson,
                        "normalized_euclidean_distance": normalized_euclidean_distance(
                            component_window,
                            reference_window,
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def aligned_reference_spectra_table(wn: np.ndarray, reference_spectra: Dict[str, np.ndarray]):
    if not reference_spectra:
        return pd.DataFrame()
    table = pd.DataFrame({"Raman Shift": wn})
    for reference_name, reference_spectrum in reference_spectra.items():
        table[reference_name] = reference_spectrum
        positive_max = np.nanmax(np.maximum(reference_spectrum, 0.0))
        table[f"{reference_name}_normalized_to_positive_max"] = (
            reference_spectrum / positive_max if positive_max > 0 else reference_spectrum
        )
    return table


def detect_major_component_peaks(S: np.ndarray, wn: np.ndarray, params: MCRParams):
    """Find the main Raman peaks of each resolved MCR-ALS component.

    Peak picking is performed on each component spectrum normalized to its own
    positive maximum. The output is intentionally compact and is used only for
    figure labels and the component-spectra CSV peak-summary columns.
    """
    records = []
    max_labels = int(max(0, params.component_peak_max_labels))
    if max_labels == 0:
        return pd.DataFrame()

    for component_index, spectrum in enumerate(S, start=1):
        raw = np.asarray(spectrum, dtype=float)
        positive_max = float(np.nanmax(raw)) if raw.size else 0.0
        normalized = raw / positive_max if positive_max > 0 else raw.copy()
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        peaks, props = find_peaks(
            normalized,
            height=float(params.component_peak_min_height_fraction),
            prominence=float(params.component_peak_min_prominence_fraction),
            distance=int(max(1, params.component_peak_min_distance_points)),
        )
        if len(peaks) == 0:
            continue

        prominences = props.get("prominences", np.zeros(len(peaks)))
        heights = props.get("peak_heights", normalized[peaks])
        selected_order = np.argsort(prominences)[::-1][:max_labels]
        for rank, order_index in enumerate(selected_order, start=1):
            peak_index = int(peaks[order_index])
            records.append(
                {
                    "component": component_index,
                    "component_label": f"MC{component_index}",
                    "peak_rank_by_prominence": rank,
                    "peak_index": peak_index,
                    "major_peak_raman_shift_cm-1": float(wn[peak_index]),
                    "major_peak_intensity": float(raw[peak_index]),
                    "major_peak_normalized_intensity": float(heights[order_index]),
                    "major_peak_prominence": float(prominences[order_index]),
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["component", "peak_rank_by_prominence"]
    ).reset_index(drop=True)


def add_major_peaks_to_spectra_table(spectra_df: pd.DataFrame, major_peaks_df: pd.DataFrame):
    """Append a compact major-peak summary section as columns in spectra_df."""
    peak_columns = [
        "Major peak component",
        "Major peak rank by prominence",
        "Major peak Raman shift (cm-1)",
        "Major peak intensity",
        "Major peak normalized intensity",
        "Major peak prominence",
    ]
    for col in peak_columns:
        spectra_df[col] = np.nan

    if major_peaks_df.empty:
        return spectra_df

    n_rows = min(len(major_peaks_df), len(spectra_df))
    source = major_peaks_df.iloc[:n_rows].reset_index(drop=True)
    spectra_df.loc[: n_rows - 1, "Major peak component"] = source["component_label"].to_numpy()
    spectra_df.loc[: n_rows - 1, "Major peak rank by prominence"] = source[
        "peak_rank_by_prominence"
    ].to_numpy()
    spectra_df.loc[: n_rows - 1, "Major peak Raman shift (cm-1)"] = source[
        "major_peak_raman_shift_cm-1"
    ].to_numpy()
    spectra_df.loc[: n_rows - 1, "Major peak intensity"] = source[
        "major_peak_intensity"
    ].to_numpy()
    spectra_df.loc[: n_rows - 1, "Major peak normalized intensity"] = source[
        "major_peak_normalized_intensity"
    ].to_numpy()
    spectra_df.loc[: n_rows - 1, "Major peak prominence"] = source[
        "major_peak_prominence"
    ].to_numpy()
    return spectra_df
def random_spectra_initialization(
    D: np.ndarray,
    n_components: int,
    random_seed: int,
):
    rng = np.random.default_rng(random_seed)
    # Literature-style random spectral-variable initialization:
    # for each spectral variable j, draw S0[:, j] within [Vj,min, Vj,max].
    # The generated spectra are then projected to a tiny positive floor because
    # the ALS routine below enforces non-negativity through NNLS.
    variable_min = np.nanmin(D, axis=0)
    variable_max = np.nanmax(D, axis=0)
    span = variable_max - variable_min
    S0 = rng.random((n_components, D.shape[1])) * span[None, :] + variable_min[None, :]
    S0 = np.maximum(S0, np.finfo(float).eps)
    return S0
def normalize_scale(C: np.ndarray, S: np.ndarray):
    C_scaled = C.copy()
    S_scaled = S.copy()
    for i in range(S.shape[0]):
        max_value = np.max(S_scaled[i])
        if max_value > 0:
            S_scaled[i] /= max_value
            C_scaled[:, i] *= max_value
    return C_scaled, S_scaled


def mcr_als_nonnegative(
    D: np.ndarray,
    n_components: int,
    params: MCRParams,
    initial_spectra: np.ndarray,
):
    S = np.asarray(initial_spectra, dtype=float).copy()
    if S.shape != (n_components, D.shape[1]):
        raise ValueError("initial_spectra has an incompatible shape.")
    if np.any(S < 0) or not np.all(np.isfinite(S)):
        raise ValueError("initial_spectra must be finite and nonnegative.")
    C = np.zeros((D.shape[0], n_components), dtype=float)
    previous_error = None
    error_history = []
    relative_change = np.nan
    converged = False

    for iteration in range(params.max_iter):
        for row in range(D.shape[0]):
            C[row, :], _ = nnls(S.T, D[row, :])

        for band in range(D.shape[1]):
            S[:, band], _ = nnls(C, D[:, band])

        C, S = normalize_scale(C, S)
        error = np.linalg.norm(D - C @ S)
        error_history.append(float(error))

        if previous_error is not None:
            relative_change = abs(previous_error - error) / max(previous_error, 1e-12)
            if relative_change < params.tol:
                converged = True
                break
        previous_error = error

    metrics = fit_metrics(D, C, S)
    metrics["n_iter"] = iteration + 1
    metrics["converged"] = converged
    metrics["final_relative_change"] = float(relative_change)
    metrics["error_history"] = error_history
    error_differences = np.diff(error_history)
    metrics["monotonic_error_decrease"] = bool(
        np.all(error_differences <= np.maximum(1e-12, 1e-10 * np.asarray(error_history[:-1])))
    )
    return C, S, metrics


def align_solution_to_reference(
    C: np.ndarray,
    S: np.ndarray,
    reference_C: np.ndarray,
    reference_S: np.ndarray,
):
    s_norms = np.linalg.norm(S, axis=1, keepdims=True)
    reference_norms = np.linalg.norm(reference_S, axis=1, keepdims=True)
    S_unit = np.divide(S, s_norms, out=np.zeros_like(S), where=s_norms > 0)
    reference_unit = np.divide(
        reference_S,
        reference_norms,
        out=np.zeros_like(reference_S),
        where=reference_norms > 0,
    )
    similarities = np.clip(S_unit @ reference_unit.T, -1.0, 1.0)
    rows, columns = linear_sum_assignment(-similarities)
    order = np.empty(S.shape[0], dtype=int)
    order[columns] = rows
    C_aligned = C[:, order]
    S_aligned = S[order, :]
    matched_spectral_cosines = similarities[order, np.arange(S.shape[0])]

    map_correlations = []
    for component in range(C.shape[1]):
        candidate = C_aligned[:, component]
        reference = reference_C[:, component]
        if np.std(candidate) <= np.finfo(float).eps or np.std(reference) <= np.finfo(float).eps:
            map_correlations.append(np.nan)
        else:
            map_correlations.append(float(np.corrcoef(candidate, reference)[0, 1]))
    return C_aligned, S_aligned, matched_spectral_cosines, np.asarray(map_correlations)
def run_initialization_family_outputs(
    D: np.ndarray,
    n_components: int,
    params: MCRParams,
):
    """Return the best solution among random spectral initializations only."""
    starts = []
    for start_index in range(params.n_random_starts):
        seed = params.random_state + start_index
        starts.append(
            (
                "random_best",
                "random",
                seed,
                random_spectra_initialization(D, n_components, seed),
            )
        )

    solutions = []
    for run_index, (output_family, method, seed, initial_spectra) in enumerate(starts, start=1):
        C, S, metrics = mcr_als_nonnegative(
            D,
            n_components,
            params,
            initial_spectra,
        )
        metrics["run"] = run_index
        metrics["initialization_family"] = output_family
        metrics["initialization_method"] = method
        metrics["initialization_seed"] = seed
        solutions.append(
            {
                "run": run_index,
                "initialization_family": output_family,
                "initialization_method": method,
                "initialization_seed": seed,
                "C": C,
                "S": S,
                "metrics": metrics,
            }
        )

    best_index = int(
        np.argmin([solution["metrics"]["error_history"][-1] for solution in solutions])
    )
    reference_C = solutions[best_index]["C"]
    reference_S = solutions[best_index]["S"]

    records = []
    for solution_index, solution in enumerate(solutions):
        C_aligned, S_aligned, spectral_cosines, map_correlations = align_solution_to_reference(
            solution["C"],
            solution["S"],
            reference_C,
            reference_S,
        )
        solution["C_aligned"] = C_aligned
        solution["S_aligned"] = S_aligned
        finite_map_correlations = map_correlations[np.isfinite(map_correlations)]
        records.append(
            {
                "run": solution["run"],
                "initialization_family": solution["initialization_family"],
                "initialization_method": solution["initialization_method"],
                "initialization_seed": solution["initialization_seed"],
                "selected_as_random_best_output": solution_index == best_index,
                "LOF_percent": solution["metrics"]["LOF_percent"],
                "explained_signal_percent": solution["metrics"]["explained_variance_percent"],
                "RMSE": solution["metrics"]["RMSE"],
                "n_iter": solution["metrics"]["n_iter"],
                "converged": solution["metrics"]["converged"],
                "monotonic_error_decrease": solution["metrics"]["monotonic_error_decrease"],
                "mean_spectral_cosine_to_order_reference": float(np.mean(spectral_cosines)),
                "minimum_spectral_cosine_to_order_reference": float(np.min(spectral_cosines)),
                "mean_map_correlation_to_order_reference": (
                    float(np.mean(finite_map_correlations))
                    if finite_map_correlations.size
                    else np.nan
                ),
                "minimum_map_correlation_to_order_reference": (
                    float(np.min(finite_map_correlations))
                    if finite_map_correlations.size
                    else np.nan
                ),
            }
        )

    stability_table = pd.DataFrame(records)
    solution = solutions[best_index]
    metrics = solution["metrics"].copy()
    metrics["output_tag"] = "random_best"
    metrics["selection_scope"] = "random_only_output"
    metrics["component_order_reference_run"] = solution["run"]
    metrics["initialization_stability"] = stability_table.copy()
    metrics["simplisma_diagnostics"] = pd.DataFrame()
    metrics["purest_row_diagnostics"] = pd.DataFrame()
    return [
        (
            "random_best",
            solution["C_aligned"],
            solution["S_aligned"],
            metrics,
        )
    ]


def fit_metrics(D: np.ndarray, C: np.ndarray, S: np.ndarray):
    residual = D - C @ S
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum(D ** 2)
    return {
        "LOF_percent": 100 * np.sqrt(ss_res / ss_tot),
        "explained_variance_percent": 100 * (1 - ss_res / ss_tot),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "residual_standard_deviation": float(np.std(residual)),
    }


def component_contributions(C: np.ndarray, labels: List[str]):
    totals = np.sum(C, axis=0)
    fractions = totals / np.sum(totals) if np.sum(totals) > 0 else np.zeros_like(totals)
    return pd.DataFrame(
        {
            "component": np.arange(1, C.shape[1] + 1),
            "assignment": labels,
            "relative_contribution_fraction": fractions,
            "relative_contribution_percent": fractions * 100,
            "total_C_signal": totals,
            "mean_C_signal": np.mean(C, axis=0),
        }
    )


def pca_rank_check(D: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(D)
    return pd.DataFrame(
        {
            "PC": np.arange(1, n_components + 1),
            "explained_variance_percent": pca.explained_variance_ratio_ * 100,
            "cumulative_explained_variance_percent": np.cumsum(pca.explained_variance_ratio_) * 100,
        }
    )


def svd_rank_check(D: np.ndarray, n_components: int):
    singular_values = np.linalg.svd(D, full_matrices=False, compute_uv=False)
    energy = singular_values ** 2
    total_energy = np.sum(energy)
    n_report = min(n_components, len(singular_values))
    records = []
    for i in range(n_report):
        residual_energy = np.sum(energy[i + 1 :])
        records.append(
            {
                "component": i + 1,
                "singular_value": singular_values[i],
                "singular_value_relative_to_first": singular_values[i] / singular_values[0],
                "explained_signal_percent": 100 * energy[i] / total_energy,
                "cumulative_explained_signal_percent": 100 * np.sum(energy[: i + 1]) / total_energy,
                "best_rank_k_LOF_percent": 100 * np.sqrt(residual_energy / total_energy),
            }
        )
    return pd.DataFrame(records)


def build_maps(C: np.ndarray, valid_mask: np.ndarray, params: MCRParams):
    C_map = C.copy()
    if params.sum_to_one_maps:
        row_sums = np.sum(C_map, axis=1, keepdims=True)
        C_map = np.divide(C_map, row_sums, out=np.zeros_like(C_map), where=row_sums > 0)

    maps = np.full((*valid_mask.shape, C.shape[1]), np.nan, dtype=float)
    for i in range(C.shape[1]):
        maps[:, :, i][valid_mask] = C_map[:, i]
    return maps


def map_for_scan_display(image_data: np.ndarray, params: MCRParams):
    """Return a display array using the corrected scan-coordinate convention.

    Raw maps are indexed as [X, Y]. The plotted vertical axis is raw X and the
    plotted horizontal axis is raw Y. Since the acquisition origin X0Y0 is at the
    bottom-right and Y increases from right to left, only the horizontal display
    axis is flipped. No transpose is applied.
    """
    display = np.asarray(image_data, dtype=float)
    if params.flip_horizontal_y_for_display:
        display = np.fliplr(display)
    return display


def display_position_from_raw_xy(x: int, y: int, map_shape: Tuple[int, int], params: MCRParams):
    """Return display row/column for a raw acquisition coordinate X/Y."""
    _, ny = map_shape
    display_row = int(x)
    display_col = int(ny - 1 - y if params.flip_horizontal_y_for_display else y)
    return display_row, display_col


def _axis_tick_values(n_points: int, step: int = 10):
    ticks = list(range(0, int(n_points), step))
    if not ticks:
        ticks = [0]
    return ticks


def apply_scan_coordinate_ticks(ax, map_shape: Tuple[int, int], params: MCRParams):
    nx, ny = map_shape.shape if hasattr(map_shape, "shape") else map_shape
    pixel = float(params.map_pixel_size_um)

    y_raw_ticks = _axis_tick_values(ny)
    if params.flip_horizontal_y_for_display:
        horizontal_pairs = sorted((ny - 1 - y, y * pixel) for y in y_raw_ticks)
    else:
        horizontal_pairs = sorted((y, y * pixel) for y in y_raw_ticks)
    x_raw_ticks = _axis_tick_values(nx)

    ax.set_xlabel(r"Y ($\mu$m)")
    ax.set_ylabel(r"X ($\mu$m)")
    ax.set_xticks([p for p, _ in horizontal_pairs])
    ax.set_xticklabels([f"{v:g}" for _, v in horizontal_pairs])
    ax.set_yticks(x_raw_ticks)
    ax.set_yticklabels([f"{x * pixel:g}" for x in x_raw_ticks])
    ax.set_xlim(-0.5, ny - 0.5)
    ax.set_ylim(-0.5, nx - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=2.5, width=0.7, pad=2)


def add_map_scalebar(ax, map_shape: Tuple[int, int], params: MCRParams):
    nx, ny = map_shape
    pixel = float(params.map_pixel_size_um)
    if pixel <= 0 or params.scalebar_um <= 0:
        return
    bar_len_px = min(params.scalebar_um / pixel, max(1.0, ny * 0.35))
    x0 = 2.0
    y0 = 2.0
    ax.plot([x0, x0 + bar_len_px], [y0, y0], color="black", linewidth=3.4, solid_capstyle="butt")
    ax.plot([x0, x0 + bar_len_px], [y0, y0], color="white", linewidth=2.0, solid_capstyle="butt")
    ax.text(
        x0 + bar_len_px / 2,
        y0 + 1.8,
        f"{params.scalebar_um:g} $\\mu$m",
        color="white",
        ha="center",
        va="bottom",
        fontsize=7,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
    )


def prepare_display_contour_map(display_map: np.ndarray, params: MCRParams):
    """Prepare an Origin-like filled contour map with optional very light smoothing.

    No upsampling is applied. Smoothing is display-only and does not change the
    exported raw maps, component scores, spectra, or MCR-ALS results.
    """
    original = np.asarray(display_map, dtype=float)
    finite_mask = np.isfinite(original)
    if not np.any(finite_mask):
        return np.ma.masked_invalid(original)

    fill_value = float(np.nanmedian(original[finite_mask]))
    contour_map = np.where(finite_mask, original, fill_value)
    sigma = float(getattr(params, "map_display_smoothing_sigma_px", 0.0))
    if sigma > 0:
        contour_map = gaussian_filter(contour_map, sigma=sigma, mode="nearest")
    return np.ma.masked_where(~finite_mask, contour_map)


def component_map_mesh(display_map: np.ndarray):
    n_rows, n_cols = display_map.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    return np.meshgrid(x, y)


def map_display_limits(display_map: np.ndarray, params: MCRParams):
    finite_values = np.asarray(display_map, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    low_pct = float(getattr(params, "map_contrast_low_percentile", 0.0))
    high_pct = float(getattr(params, "map_contrast_high_percentile", 100.0))
    vmin = float(np.nanpercentile(finite_values, low_pct))
    vmax = float(np.nanpercentile(finite_values, high_pct))
    if np.nanmin(finite_values) >= 0 and low_pct <= 0:
        vmin = 0.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    return vmin, vmax


def black_to_bright_colormap(color: str, name: str):
    return LinearSegmentedColormap.from_list(
        name,
        ["#000000", "#17262B", color],
        N=256,
    )


def band_max(spectrum: np.ndarray, wn: np.ndarray, low: float, high: float):
    mask = (wn >= low) & (wn <= high)
    return float(np.max(spectrum[mask])) if np.any(mask) else 0.0


def component_band_indicators(S: np.ndarray, wn: np.ndarray):
    records = []
    for i, s in enumerate(S):
        phosphate = band_max(s, wn, 940, 980)
        carbonate = band_max(s, wn, 1050, 1095)
        collagen = (
            band_max(s, wn, 850, 880)
            + band_max(s, wn, 1230, 1285)
            + band_max(s, wn, 1430, 1485)
            + band_max(s, wn, 1640, 1685)
        )
        records.append(
            {
                "component": i + 1,
                "phosphate_940_980": phosphate,
                "carbonate_1050_1095": carbonate,
                "collagen_score": collagen,
                "carbonate_to_phosphate": carbonate / (phosphate + 1e-12),
            }
        )

    return pd.DataFrame(records)


def rank_selection_summary_table(
    rank_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    svd_df: pd.DataFrame,
    params: MCRParams,
):
    """One compact source table for component-number selection."""
    if rank_df.empty:
        return pd.DataFrame()

    rows = []
    ranked = rank_df.sort_values("n_components").reset_index(drop=True)
    previous_lof = None
    for _, row in ranked.iterrows():
        k = int(row["n_components"])
        pca_match = pca_df.loc[pca_df["PC"] == k]
        svd_match = svd_df.loc[svd_df["component"] == k]
        lof = float(row["LOF_percent"])
        rows.append(
            {
                "Sample": params.sample_name,
                "k": k,
                "LOF decrease from previous k (%)": (
                    np.nan if previous_lof is None else previous_lof - lof
                ),
                "Centered PCA cumulative EV (%)": (
                    float(pca_match["cumulative_explained_variance_percent"].iloc[0])
                    if not pca_match.empty else np.nan
                ),
                "Uncentered SVD best-rank-k LOF (%)": (
                    float(svd_match["best_rank_k_LOF_percent"].iloc[0])
                    if not svd_match.empty else np.nan
                ),
                "LOF (%)": lof,
                "EV (%)": float(row["explained_variance_percent"]),
                "RMSE": float(row["RMSE"]),
                "Residual SD": float(row["residual_standard_deviation"]),
            }
        )
        previous_lof = lof
    return pd.DataFrame(rows)


def plot_rank_selection_summary(rank_summary_df: pd.DataFrame, save_path: str):
    """One PNG for deciding the number of MCR-ALS components."""
    if rank_summary_df.empty:
        return

    k = rank_summary_df["k"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    axes[0, 0].plot(k, rank_summary_df["LOF (%)"], marker="o", color="#3C5488", label="MCR-ALS LOF")
    axes[0, 0].plot(
        k,
        rank_summary_df["Uncentered SVD best-rank-k LOF (%)"],
        marker="s",
        color="#E64B35",
        linestyle="--",
        label="Best rank-k SVD LOF",
    )
    axes[0, 0].set_title("Lack of fit versus rank")
    axes[0, 0].set_xlabel("Number of components, k")
    axes[0, 0].set_ylabel("LOF (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(k, rank_summary_df["EV (%)"], marker="o", color="#00A087")
    axes[0, 1].set_title("MCR-ALS explained variance")
    axes[0, 1].set_xlabel("Number of components, k")
    axes[0, 1].set_ylabel("EV (%)")
    axes[0, 1].grid(True, alpha=0.25)

    decrease = rank_summary_df["LOF decrease from previous k (%)"].fillna(0.0)
    axes[1, 0].bar(k, decrease, color="#4DBBD5", alpha=0.85)
    axes[1, 0].set_title("Incremental LOF decrease")
    axes[1, 0].set_xlabel("Number of components, k")
    axes[1, 0].set_ylabel("LOF decrease from previous k (%)")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    axes[1, 1].plot(
        k,
        rank_summary_df["Centered PCA cumulative EV (%)"],
        marker="o",
        color="#8491B4",
    )
    axes[1, 1].set_title("Centered PCA cumulative explained variance")
    axes[1, 1].set_xlabel("Principal components, k")
    axes[1, 1].set_ylabel("Cumulative EV (%)")
    axes[1, 1].grid(True, alpha=0.25)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
def plot_component_reference_overlay(
    S: np.ndarray,
    wn: np.ndarray,
    labels: List[str],
    reference_spectra: Dict[str, np.ndarray],
    params: MCRParams,
    save_path: str,
):
    if not reference_spectra:
        return

    major_peaks_df = detect_major_component_peaks(S, wn, params)
    fig, axes = plt.subplots(S.shape[0], 1, figsize=(10, 2.9 * S.shape[0]), sharex=True)
    if S.shape[0] == 1:
        axes = [axes]

    reference_styles = {
        "HA_reference": {"color": "tab:orange", "linestyle": "--"},
        "collagen_reference": {"color": "tab:green", "linestyle": ":"},
    }
    for i, ax in enumerate(axes):
        component = S[i]
        component_max = np.max(component)
        component_plot = component / component_max if component_max > 0 else component
        ax.plot(wn, component_plot, color="black", linewidth=1.6, label=labels[i])

        for reference_name, reference_spectrum in reference_spectra.items():
            positive_max = np.nanmax(np.maximum(reference_spectrum, 0.0))
            reference_plot = (
                reference_spectrum / positive_max if positive_max > 0 else reference_spectrum
            )
            style = reference_styles.get(reference_name, {"linestyle": "--"})
            ax.plot(
                wn,
                reference_plot,
                linewidth=1.15,
                alpha=0.85,
                label=reference_name,
                **style,
            )

        component_peaks = major_peaks_df[
            major_peaks_df["component"] == i + 1
        ].sort_values("major_peak_raman_shift_cm-1") if not major_peaks_df.empty else pd.DataFrame()
        for _, peak_row in component_peaks.iterrows():
            x = float(peak_row["major_peak_raman_shift_cm-1"])
            peak_index = int(peak_row["peak_index"])
            y = float(component_plot[peak_index])
            ax.plot(x, y, marker="o", markersize=3.0, color="black", zorder=4)
            ax.annotate(
                f"{x:.0f}",
                xy=(x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="black",
            )

        ax.set_title(f"{labels[i]} versus HA/collagen references")
        ax.set_ylabel("Normalized intensity")
        ax.set_ylim(-0.05, 1.18)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        for peak in [430, 590, 960, 1070, 1245, 1450, 1660]:
            ax.axvline(peak, color="0.82", linewidth=0.7, linestyle="--")

    axes[-1].set_xlabel(r"Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
def publication_map_rows(maps: np.ndarray, params: MCRParams):
    rows = []
    nx, ny, n_components = maps.shape
    for x in range(nx):
        for y in range(ny):
            display_row, display_col = display_position_from_raw_xy(x, y, (nx, ny), params)
            row = {
                "Acquisition_X_raw_vertical_axis": x,
                "Acquisition_Y_raw_horizontal_axis": y,
                "X_um_from_origin_vertical": x * params.map_pixel_size_um,
                "Y_um_from_origin_horizontal": y * params.map_pixel_size_um,
                "Display_row_for_X_axis": display_row,
                "Display_column_for_Y_axis": display_col,
                "Origin_note": "X0Y0 is the bottom-right pixel in the displayed map",
            }
            for i in range(n_components):
                row[f"Component {i + 1}"] = maps[x, y, i]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_component_maps(maps: np.ndarray, labels: List[str], save_path: str, params: MCRParams):
    n_components = maps.shape[2]
    fig, axes = plt.subplots(
        1,
        n_components,
        figsize=(3.25 * n_components, 3.35),
        constrained_layout=True,
    )
    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        display_map = map_for_scan_display(maps[:, :, i], params)
        contour_map = prepare_display_contour_map(display_map, params)
        mesh_x, mesh_y = component_map_mesh(display_map)
        color = params.component_map_colors[i % len(params.component_map_colors)]
        cmap = black_to_bright_colormap(color, f"black_to_component_{i + 1}")
        cmap.set_bad("white")
        vmin, vmax = map_display_limits(display_map, params)
        levels = np.linspace(vmin, vmax, int(params.contour_levels))

        im = ax.contourf(
            mesh_x,
            mesh_y,
            contour_map,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend="max",
            antialiased=False,
        )

        ax.set_title(f"MC{i + 1}", fontsize=8, pad=3)
        apply_scan_coordinate_ticks(ax, maps[:, :, i].shape, params)
        add_map_scalebar(ax, maps[:, :, i].shape, params)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label(f"MC{i + 1} intensity", fontsize=7)
        cbar.ax.tick_params(labelsize=7, length=2.0, width=0.6)

    fig.savefig(save_path, dpi=600)
    plt.close(fig)


def plot_mean_fit(D: np.ndarray, C: np.ndarray, S: np.ndarray, wn: np.ndarray, save_path: str):
    mean_original = np.mean(D, axis=0)
    mean_reconstructed = np.mean(C @ S, axis=0)
    residual = mean_original - mean_reconstructed

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wn, mean_original, color="black", linewidth=1.5, label="Mean original")
    ax.plot(wn, mean_reconstructed, color="red", linestyle="--", linewidth=1.4, label="Mean reconstructed")
    ax.plot(wn, residual, color="0.45", linewidth=1.0, label="Residual")
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Area-normalized intensity")
    ax.set_title("Mean spectrum reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def export_model(
    params: MCRParams,
    k: int,
    wn: np.ndarray,
    valid_mask: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    maps: np.ndarray,
    labels: List[str],
    scores: pd.DataFrame,
    metrics: dict,
    reference_spectra: Dict[str, np.ndarray],
):
    """Export only the manuscript-relevant MCR-ALS outputs requested by the user."""
    output_tag = metrics.get("output_tag", "")
    output_suffix = f"_{output_tag}" if output_tag else ""
    model_dir = os.path.join(params.save_dir, f"{params.sample_name}_{k}components{output_suffix}")
    os.makedirs(model_dir, exist_ok=True)
    base = os.path.join(model_dir, f"{params.sample_name}_{k}components")

    # 1) Relative concentrations/scores of MCR components.
    relative_concentration_df = component_contributions(C, labels).rename(
        columns={
            "assignment": "component_label",
            "relative_contribution_fraction": "relative_concentration_fraction",
            "relative_contribution_percent": "relative_concentration_percent",
            "total_C_signal": "total_component_score",
            "mean_C_signal": "mean_component_score",
        }
    )
    relative_concentration_df["interpretation_note"] = (
        "Relative concentration is calculated from the summed MCR-ALS C scores; "
        "it is not an absolute chemical mass or volume fraction."
    )
    relative_concentration_df.to_csv(
        f"{base}_relative_concentrations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2) Component spectra: Raman shift and intensity.
    spectra_df = pd.DataFrame({"Raman Shift (cm-1)": wn})
    for i in range(k):
        positive_max = np.max(S[i]) if np.max(S[i]) > 0 else 0.0
        spectra_df[f"MC{i + 1} intensity"] = S[i]
        spectra_df[f"MC{i + 1} normalized intensity"] = S[i] / positive_max if positive_max > 0 else S[i]
    major_peaks_df = detect_major_component_peaks(S, wn, params)
    spectra_df = add_major_peaks_to_spectra_table(spectra_df, major_peaks_df)
    spectra_df.to_csv(
        f"{base}_component_spectra.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3) Component map source data.
    publication_map_rows(maps, params).to_csv(
        f"{base}_component_maps_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4) Full-spectrum similarity to HA and collagen references only.
    reference_similarity_df = reference_similarity_table(S, wn, reference_spectra, params)
    if not reference_similarity_df.empty:
        reference_similarity_df = reference_similarity_df[
            reference_similarity_df["window"] == "full_400_1800"
        ].copy()
        reference_similarity_df.to_csv(
            f"{base}_reference_similarity_full_400_1800.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # 5) Required figures only.
    plot_component_reference_overlay(
        S,
        wn,
        labels,
        reference_spectra,
        params,
        f"{base}_spectra_with_HA_collagen_references.png",
    )
    plot_component_maps(
        maps,
        labels,
        f"{base}_maps.png",
        params,
    )


def main():
    params = MCRParams()
    os.makedirs(params.save_dir, exist_ok=True)

    wn, valid_mask, D = load_preprocessed_data(params)
    print(f"Loaded spectra matrix D: {D.shape[0]} pixels x {D.shape[1]} bands")
    print(f"Excluded pixels from preprocessing: {valid_mask.size - np.sum(valid_mask)}")
    reference_spectra = load_reference_spectra(params, wn)
    print(
        "Loaded reference spectra: "
        + ", ".join(reference_spectra.keys())
        + f" from {params.reference_spectra_path}"
    )

    pca_df = pca_rank_check(D, max(params.component_numbers))
    svd_df = svd_rank_check(D, params.svd_components_to_report)
    rank_records = []

    for k in params.component_numbers:
        print(f"\nRunning MCR-ALS for k={k}")
        output_solutions = run_initialization_family_outputs(D, k, params)
        for output_tag, C, S, metrics in output_solutions:
            labels = [f"MC{i + 1}" for i in range(k)]
            scores = component_band_indicators(S, wn)
            maps = build_maps(C, valid_mask, params)
            contribution_df = component_contributions(C, labels)
            export_model(
                params,
                k,
                wn,
                valid_mask,
                D,
                C,
                S,
                maps,
                labels,
                scores,
                metrics,
                reference_spectra,
            )

            rank_row = {
                "n_components": k,
                "initialization_output": output_tag,
                "initialization_method": metrics["initialization_method"],
                "initialization_seed": metrics["initialization_seed"],
                "component_order_reference_run": metrics["component_order_reference_run"],
                "LOF_percent": metrics["LOF_percent"],
                "explained_variance_percent": metrics["explained_variance_percent"],
                "n_iter": metrics["n_iter"],
                "converged": metrics["converged"],
                "final_relative_change": metrics["final_relative_change"],
                "monotonic_error_decrease": metrics["monotonic_error_decrease"],
                "RMSE": metrics["RMSE"],
                "residual_standard_deviation": metrics["residual_standard_deviation"],
                "best_rank_k_SVD_LOF_percent": svd_df.loc[
                    svd_df["component"] == k, "best_rank_k_LOF_percent"
                ].iloc[0],
            }
            for _, row in contribution_df.iterrows():
                comp = int(row["component"])
                rank_row[f"Component {comp} contribution percent"] = row["relative_contribution_percent"]
                rank_row[f"Component {comp} assignment"] = row["assignment"]
            rank_records.append(rank_row)

            print(
                f"k={k}, {output_tag}: "
                f"LOF={metrics['LOF_percent']:.3f}%, "
                f"EV={metrics['explained_variance_percent']:.3f}%, "
                f"method={metrics['initialization_method']}, "
                f"seed={metrics['initialization_seed']}"
            )
            for i, label in enumerate(labels, start=1):
                percent = contribution_df.loc[
                    contribution_df["component"] == i,
                    "relative_contribution_percent",
                ].iloc[0]
                print(f"  Component {i}: {label}; relative contribution = {percent:.2f}%")

    rank_df = pd.DataFrame(rank_records)
    rank_summary_df = rank_selection_summary_table(rank_df, pca_df, svd_df, params)
    rank_summary_df.to_csv(
        os.path.join(params.save_dir, f"{params.sample_name}_rank_selection_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    plot_rank_selection_summary(
        rank_summary_df,
        os.path.join(params.save_dir, f"{params.sample_name}_rank_selection_summary.png"),
    )

    print(f"\nAll results exported to: {params.save_dir}")


if __name__ == "__main__":
    main()
