"""
Load an analysis object from SEEKR2 and perform an implied timescales
analysis.
"""

import os
import argparse
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seekr2.modules.common_base as base
from seekr2.modules.common_analyze import DEFAULT_IMAGE_DIR
import seekr2.analyze as analyze

from . import chapman_kolmogorov_test

IMPLIED_TIME_DIR = "implied_timescales"

def compute_implied_timescales(transition_matrix, lag):
    """Compute implied timescales from a transition matrix at a given lag.
    
    Uses the magnitude of each eigenvalue to compute decorrelation
    timescales. Performs numerical health checks on the transition matrix
    and eigenvalues, issuing warnings when results may be unreliable.
    
    Parameters
    ----------
    transition_matrix : numpy.ndarray
        Row-stochastic transition matrix of shape (M, M).
    lag : int
        The lag time used to estimate this transition matrix.
    
    Returns
    -------
    timescales : numpy.ndarray
        Array of M-1 implied timescales (excluding the stationary mode),
        sorted from slowest to fastest. Units match the lag units.
        Modes with |lambda| >= 1, |lambda| == 0, or below the numerical
        noise floor are assigned np.nan.
    eigenvalues_mag : numpy.ndarray
        Sorted eigenvalue magnitudes (all M, descending).
    
    Raises
    ------
    ValueError
        If the transition matrix contains NaN or Inf values.
    """
    # --- Pre-checks on the transition matrix ---
    if np.any(np.isnan(transition_matrix)) or np.any(np.isinf(transition_matrix)):
        raise ValueError(
            f"Transition matrix contains NaN or Inf at lag {lag}.")
    
    # Check for negative entries
    if np.any(transition_matrix < -1e-12):
        warnings.warn(
            f"Lag {lag}: Transition matrix has negative entries "
            f"(min={transition_matrix.min():.2e}). "
            f"This may indicate insufficient data or a normalization error.",
            RuntimeWarning)
    
    # Check row sums
    row_sums = transition_matrix.sum(axis=1)
    row_sum_deviation = np.max(np.abs(row_sums - 1.0))
    if row_sum_deviation > 1e-6:
        warnings.warn(
            f"Lag {lag}: Transition matrix rows do not sum to 1.0 "
            f"(max deviation={row_sum_deviation:.2e}). "
            f"Eigenvalues may be unreliable.",
            RuntimeWarning)
    
    # Check condition number
    cond = np.linalg.cond(transition_matrix)
    if cond > 1e12:
        warnings.warn(
            f"Lag {lag}: Transition matrix is ill-conditioned "
            f"(condition number={cond:.2e}). "
            f"Eigenvalues may have low numerical precision.",
            RuntimeWarning)
    
    # --- Compute eigenvalues ---
    eigenvalues = np.linalg.eigvals(transition_matrix)
    eigenvalues_mag = np.sort(np.abs(eigenvalues))[::-1]
    
    # --- Post-checks on eigenvalues ---
    # Check that the largest eigenvalue is close to 1.0
    if abs(eigenvalues_mag[0] - 1.0) > 1e-4:
        warnings.warn(
            f"Lag {lag}: Largest eigenvalue magnitude is "
            f"{eigenvalues_mag[0]:.6f}, expected ~1.0. "
            f"The matrix may not be properly stochastic.",
            RuntimeWarning)
    
    # Check for eigenvalues exceeding 1.0
    if eigenvalues_mag[0] > 1.0 + 1e-8:
        warnings.warn(
            f"Lag {lag}: Eigenvalue magnitude {eigenvalues_mag[0]:.6f} > 1.0. "
            f"This is unphysical for a stochastic matrix and indicates "
            f"numerical issues.",
            RuntimeWarning)
    
    # Noise floor: eigenvalues below this are at machine precision
    machine_eps = np.finfo(transition_matrix.dtype).eps
    noise_floor = np.sqrt(machine_eps) * max(transition_matrix.shape)
    
    # Skip the first eigenvalue (~1.0, stationary). Compute timescales
    # for the remaining M-1 eigenvalues.
    timescales = np.full(len(eigenvalues_mag) - 1, np.nan)
    for i, eigval_mag in enumerate(eigenvalues_mag[1:]):
        if eigval_mag < noise_floor:
            warnings.warn(
                f"Lag {lag}: Eigenvalue {i+2} has magnitude "
                f"{eigval_mag:.2e}, which is below the numerical noise "
                f"floor ({noise_floor:.2e}). Its timescale is unreliable "
                f"and will be set to NaN.",
                RuntimeWarning)
            continue
        if eigval_mag > 0.0 and eigval_mag < 1.0:
            timescales[i] = -lag / np.log(eigval_mag)
    
    return timescales, eigenvalues_mag


def bootstrap_implied_timescales(segments, ordered_keys, lag_values,
                                  n_bootstrap=10):
    """Bootstrap error estimates for implied timescales.
    
    Resamples trajectory segments with replacement (preserving temporal
    structure within each segment). For each resample, re-estimates the
    transition matrix at each lag and computes implied timescales.
    
    Parameters
    ----------
    segments : list of list of int
        Trajectory segments from parse_transition_sequence.
    ordered_keys : list of int
        Sorted milestone alias indices.
    lag_values : array-like of int
        Lag values to test.
    n_bootstrap : int
        Number of bootstrap resamples.
    
    Returns
    -------
    boot_errors : dict
        Keys are lag values. Values are dicts with:
        - 'timescales_std': numpy.ndarray of shape (M-1,) or None
        - 'timescales_low': numpy.ndarray (2.5th percentile)
        - 'timescales_high': numpy.ndarray (97.5th percentile)
    """
    M = len(ordered_keys)
    n_timescales = M - 1
    rng = np.random.default_rng()
    
    # Collect bootstrap timescales: {lag: list of arrays}
    boot_samples = {lag: [] for lag in lag_values}
    
    for segment in segments:
        n_seg = len(segment)
        for _ in range(n_bootstrap):
            # Resample positions within the segment with replacement
            indices = rng.choice(n_seg, size=n_seg, replace=True)
            resampled = [segment[idx] for idx in indices]
            
            for lag in lag_values:
                T_lag, total = chapman_kolmogorov_test\
                    .estimate_transition_matrix_at_lag(
                        [resampled], ordered_keys, lag)
                if T_lag is not None:
                    ts, _ = compute_implied_timescales(T_lag, lag)
                    boot_samples[lag].append(ts)
                else:
                    boot_samples[lag].append(
                        np.full(n_timescales, np.nan))

    boot_errors = {}
    for lag in lag_values:
        samples_array = np.array(boot_samples[lag])  # (n_bootstrap, M-1)
        if len(samples_array) > 1:
            boot_errors[lag] = {
                'timescales_std': np.nanstd(samples_array, axis=0),
                'timescales_low': np.nanpercentile(samples_array, 2.5,
                                                   axis=0),
                'timescales_high': np.nanpercentile(samples_array, 97.5,
                                                    axis=0),
            }
        else:
            boot_errors[lag] = {
                'timescales_std': None,
                'timescales_low': None,
                'timescales_high': None,
            }
    return boot_errors


def run_implied_timescales_test(segments, ordered_keys, lag_values,
                                anchor_transition_matrix=None,
                                n_bootstrap=10):
    """Run the implied timescales analysis for a single anchor.
    
    For each lag time, re-estimates the transition matrix from the raw
    trajectory data and computes eigenvalues to obtain implied timescales.
    Optionally also computes "predicted" timescales from T(1)^lag as a
    convergence reference.
    
    Parameters
    ----------
    segments : list of list of int
        Trajectory segments from parse_transition_sequence.
    ordered_keys : list of int
        Sorted milestone alias indices.
    lag_values : array-like of int
        Lag values to test (in units of bounce events).
    anchor_transition_matrix : numpy.ndarray or None
        The lag-1 transition matrix. If provided, predicted timescales
        from T(1)^lag are computed as a reference for convergence.
    n_bootstrap : int
        Number of bootstrap resamples for error estimation. Set to 0 to
        skip. Default is 10.
    
    Returns
    -------
    results : dict
        Keys are lag values. Values are dicts with:
        - 'timescales': numpy.ndarray of shape (M-1,) or None
        - 'eigenvalues': numpy.ndarray of shape (M,) or None
        - 'total_transitions': int
        - 'timescales_std': numpy.ndarray or None (bootstrap std)
        - 'timescales_low': numpy.ndarray or None (bootstrap 2.5th pctile)
        - 'timescales_high': numpy.ndarray or None (bootstrap 97.5th pctile)
        - 'predicted_timescales': numpy.ndarray of shape (M-1,) or None
    """
    M = len(ordered_keys)
    results = {}
    
    # Compute predicted timescales from T(1) eigenvalues analytically.
    # Since T(1)^lag has eigenvalues lambda^lag, the implied timescale is:
    #   t_i = -lag / ln(|lambda_i|^lag) = -lag / (lag * ln|lambda_i|)
    #       = -1 / ln(|lambda_i|)
    # This is constant across all lag values — no matrix power needed,
    # avoiding numerical instability from raising small eigenvalues to
    # large powers.
    predicted_timescales = None
    if anchor_transition_matrix is not None:
        eigvals_base = np.linalg.eigvals(anchor_transition_matrix)
        eigvals_mag = np.sort(np.abs(eigvals_base))[::-1]
        # Skip the stationary eigenvalue (~1.0)
        predicted_timescales = np.full(len(eigvals_mag) - 1, np.nan)
        for i, eigval_mag in enumerate(eigvals_mag[1:]):
            if eigval_mag > 0.0 and eigval_mag < 1.0:
                predicted_timescales[i] = -1.0 / np.log(eigval_mag)
        #print("Predicted timescales from T(1) eigenvalues:",
        #      predicted_timescales)
    
    for lag in lag_values:
        transition_matrix_at_lag, total_transitions = chapman_kolmogorov_test\
            .estimate_transition_matrix_at_lag(
                segments, ordered_keys, lag)
        
        if transition_matrix_at_lag is not None:
            timescales, eigenvalues = compute_implied_timescales(
                transition_matrix_at_lag, lag)
            results[lag] = {
                'timescales': timescales,
                'eigenvalues': eigenvalues,
                'total_transitions': total_transitions,
                'timescales_std': None,
                'timescales_low': None,
                'timescales_high': None,
                'predicted_timescales': predicted_timescales,
            }
        else:
            results[lag] = {
                'timescales': None,
                'eigenvalues': None,
                'total_transitions': 0,
                'timescales_std': None,
                'timescales_low': None,
                'timescales_high': None,
                'predicted_timescales': predicted_timescales,
            }

    """
    # Print the timescales for each lag
    print("Lag time (bounces) | Total Transitions | Implied Timescales")
    for lag in lag_values:
        res = results[lag]
        ts_str = ", ".join(f"{ts:.2f}" if ts is not None else "None"
                           for ts in res['timescales'])
        # also include the predicted timescales if available
        if res['predicted_timescales'] is not None:
            pred_str = ", ".join(f"{ts:.2f}" for ts in res['predicted_timescales'])
            ts_str += f" | Predicted: {pred_str}"
        print(f"{lag:17d} | {res['total_transitions']:17d} | {ts_str}")
    """
    
    # Bootstrap error estimation
    if n_bootstrap > 0 and sum(len(seg) for seg in segments) > 1:
        print(f"  Running {n_bootstrap} bootstrap resamples...")
        boot_errors = bootstrap_implied_timescales(
            segments, ordered_keys, lag_values, n_bootstrap)
        for lag in lag_values:
            if lag in boot_errors:
                results[lag]['timescales_std'] = \
                    boot_errors[lag]['timescales_std']
                results[lag]['timescales_low'] = \
                    boot_errors[lag]['timescales_low']
                results[lag]['timescales_high'] = \
                    boot_errors[lag]['timescales_high']
    
    return results

def plot_implied_timescales(lag_values, results, alpha, ordered_keys,
                           plot_dir, dpi=200):
    """Plot implied timescales vs. lag time for a single anchor.
    
    Creates a log-log plot with one line per non-trivial eigenmode.
    A diagonal reference line t=tau is drawn to indicate the boundary
    below which timescales are not resolved.
    
    Parameters
    ----------
    lag_values : array-like of int
        The lag values tested.
    results : dict
        Output from run_implied_timescales_test.
    alpha : int
        Anchor index (for labeling).
    ordered_keys : list of int
        Sorted milestone alias indices.
    plot_dir : str or None
        Directory to save the plot. If None, plot is displayed instead.
    dpi : int
        Resolution for saved plots.
    """
    M = len(ordered_keys)
    n_timescales = M - 1
    if n_timescales == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Collect timescale data per eigenmode
    for k in range(n_timescales):
        lags_k = []
        ts_k = []
        ts_std_k = []
        has_bootstrap = False
        
        for lag in lag_values:
            res = results.get(lag)
            if res is None or res['timescales'] is None:
                continue
            if k < len(res['timescales']):
                val = res['timescales'][k]
                if not np.isnan(val):
                    lags_k.append(lag)
                    ts_k.append(val)
                    if res['timescales_std'] is not None:
                        ts_std_k.append(res['timescales_std'][k])
                        has_bootstrap = True
                    else:
                        ts_std_k.append(np.nan)
        
        if len(lags_k) == 0:
            continue
        
        label = f"$t_{{{k+2}}}$"
        lags_arr = np.array(lags_k)
        ts_arr = np.array(ts_k)
        line, = ax.plot(lags_arr, ts_arr, marker='.', markersize=3,
                        linewidth=1.2, label=label)
        
        if has_bootstrap:
            ts_std_arr = np.array(ts_std_k)
            valid = ~np.isnan(ts_std_arr)
            if np.any(valid):
                ax.fill_between(lags_arr[valid],
                                ts_arr[valid] - ts_std_arr[valid],
                                ts_arr[valid] + ts_std_arr[valid],
                                alpha=0.2, color=line.get_color())
    
    # Plot predicted timescales from T(1)^lag as dashed reference lines
    has_predicted = any(
        results.get(lag, {}).get('predicted_timescales') is not None
        for lag in lag_values)
    if has_predicted:
        for k in range(n_timescales):
            lags_pred = []
            ts_pred = []
            for lag in lag_values:
                res = results.get(lag)
                if res is None or res.get('predicted_timescales') is None:
                    continue
                if k < len(res['predicted_timescales']):
                    val = res['predicted_timescales'][k]
                    if not np.isnan(val):
                        lags_pred.append(lag)
                        ts_pred.append(val)
            if len(lags_pred) > 0:
                label_pred = r"$T(1)^{\tau}$" if k == 0 else None
                ax.plot(lags_pred, ts_pred, '--', linewidth=1,
                        color='red', alpha=0.5, label=label_pred)
    
    # Reference line: t = tau (timescales below this are not resolved)
    all_lags = np.array(sorted(lag_values))
    ax.plot(all_lags, all_lags, '--', color='gray', linewidth=1,
            label=r'$t = \tau$', zorder=0)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Lag time $\tau$ (bounce events)')
    ax.set_ylabel('Implied timescale (bounce events)')
    ax.set_title(f'Implied Timescales: Anchor {alpha}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    if plot_dir is not None:
        filepath = os.path.join(
            plot_dir, f"implied_timescales_anchor_{alpha}.png")
        plt.savefig(filepath, dpi=dpi)
        print(f"  Saved plot: {filepath}")
    else:
        plt.show()
    plt.close(fig)


def implied_timescales_analysis(model, analysis, plot_dir=None, max_lag=1000, dpi=200,
                                n_bootstrap=10):
    """
    Perform an implied timescales analysis for a SEEKR2 model and analysis.
    This will need to be done anchor by anchor.
    
    Parameters
    ----------
    model : seekr2.modules.common_base.Model
        The SEEKR2 model object.
    analysis : seekr2.analyze.Analyze
        The SEEKR2 analysis object containing transition matrices and other
        results.
    plot_dir : str or None
        Directory to save plots. If None, plots will not be saved.
    max_lag : int
        Maximum lag time (in units of bounce events) to test. Default is 1000.
    dpi : int
        DPI (dots per inch) for saved plots. Default is 200.
    n_bootstrap : int
        Number of bootstrap resamples for error estimation. Set to 0 to skip.
        Default is 10.
    """
    for alpha, anchor in enumerate(model.anchors):
        print(f"Performing implied timescales analysis for anchor: {alpha}")
        if anchor.bulkstate:
            continue
        anchor_stats = analysis.anchor_stats_list[alpha]
        # The trajectory information transition-by-transition.
        # existing_lines is a list, whose entries might be a string
        # containing "NEW_SWARM", or might be a list of length 3
        # where the first entry is the destination boundary alias index,
        # the second entry is a bounce index (count), and the third entry
        # is the time of the transition.
        # Parse the trajectory into segments
        sequence_of_transitions = anchor_stats.existing_lines
        result = chapman_kolmogorov_test.make_anchor_transition_matrix(
            anchor, anchor_stats)
        if result is None:
            print(f"Anchor {alpha}: No transition data, skipping.")
            continue
        anchor_transition_matrix, ordered_keys = result
        segments = chapman_kolmogorov_test.parse_transition_sequence(
            sequence_of_transitions, ordered_keys)
        
        if len(segments) == 0:
            print(f"Anchor {alpha}: No trajectory segments, skipping.")
            continue
        
        total_bounces = sum(len(seg) for seg in segments)
        print(f"Anchor {alpha}: {len(ordered_keys)} milestones, "
              f"{len(segments)} segments, {total_bounces} total bounces")
        
        # Determine lag range based on available data
        effective_max_lag = min(max_lag, total_bounces // 2)
        if effective_max_lag < 2:
            print(f"Anchor {alpha}: Not enough bounces for implied "
                  f"timescales analysis, skipping.")
            continue
        
        lag_values = np.unique(np.logspace(
            0, np.log10(effective_max_lag), num=100, dtype=int))
        
        # Run implied timescales test
        implied_timescales_results = run_implied_timescales_test(
            segments, ordered_keys, lag_values,
            anchor_transition_matrix=anchor_transition_matrix,
            n_bootstrap=n_bootstrap)
        
        # Plot
        plot_implied_timescales(
            lag_values, implied_timescales_results, alpha,
            ordered_keys, plot_dir, dpi=dpi)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-n", "--n_bootstrap", dest="n_bootstrap", default=10, type=int,
        help="Number of bootstrap resamples for error estimation. "
        "Set to 0 to skip. Default: 10")
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    n_bootstrap = args["n_bootstrap"]
    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, DEFAULT_IMAGE_DIR, 
                            IMPLIED_TIME_DIR)
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {plot_dir}")
    analysis = analyze.analyze(model, num_error_samples=0, skip_checks=True)
    implied_timescales_analysis(model, analysis, plot_dir, max_lag=1000, dpi=200,
                                n_bootstrap=n_bootstrap)