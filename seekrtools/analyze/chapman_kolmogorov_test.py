"""
Load an analysis object from SEEKR2, and perform a Chapman-Kolmogorov 
test to check the Markovianity of the model. This is different from a 
typical Markov state model, because SEEKR2 uses Markovian milestoning with
Voronoi Tessellations (MMVT), which means that a trajectory is constrained
to remain within a particular Voronoi cell the entire time. Nevertheless,
we can use the trajectory data stored within the analysis object for each
anchor. 

This is tricky because the long trajectories are compute anchor-by-anchor,
so we will need to look at the long-timescale transitions anchor-by-anchor.
"""

import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seekr2.modules.common_base as base
from seekr2.modules.common_analyze import DEFAULT_IMAGE_DIR
import seekr2.analyze as analyze

CK_DIR = "ck_test_results"

def make_anchor_transition_matrix(anchor, anchor_stats):
    """
    For a given anchor, make a transition matrix that represents transitions
    between milestones available within the anchor.
    """
    # Construct a new sequence of transitions from a Markov matrix made
    # from the transitions within the anchor statistics
    ordered_keys = sorted(anchor_stats.N_alpha_beta.keys())
    if len(ordered_keys) == 0:
        # No keys - meaningless - skip
        return None
    
    key_indices = np.arange(len(ordered_keys))
    # Find sequence length by the number of transitions observed in
    #  the original anchor statistics
    sequence_length = 0
    avg_times_alpha_beta = {}
    for key in ordered_keys:
        sequence_length += anchor_stats.N_alpha_beta[key]
    
    for key in ordered_keys:
        avg_times_alpha_beta[key] = anchor_stats.T_alpha_total / \
            sequence_length \
            if sequence_length > 0 else 0.0
    
    # For starting probs
    count_vector = np.zeros(len(ordered_keys))
    # For transition probs
    count_matrix = np.zeros((len(ordered_keys), len(ordered_keys)))
    avg_times_i_j = {}
    
    for key_index in key_indices:
        for key_index2 in key_indices:
            if key_index == key_index2:
                continue # For now
            key = ordered_keys[key_index]
            key2 = ordered_keys[key_index2]
            N_i_j_alpha = anchor_stats.N_i_j_alpha[(key, key2)]
            count_matrix[key_index, key_index2] = N_i_j_alpha

        avg_times_i_j[key_index] = anchor_stats.R_i_alpha_total[key] / \
            count_matrix[key_index,:].sum() \
            if count_matrix[key_index,:].sum() > 0 else 0.0
        
    # NOTE: this is correct: N_i_j_alpha determines how many transitions
    #  to a different milestone, and N_alpha_beta contains all bounces.
    #  therefore, the value of the main diagonal is found by: 
    #  N_alpha_beta[alpha] - sum_over_j(N_i_j_alpha)
    for key_index in key_indices:
        row_sum = np.sum(count_matrix[:, key_index])
        count_matrix[key_index, key_index] \
            = anchor_stats.N_alpha_beta[ordered_keys[key_index]] - row_sum
        count_vector[key_index] = anchor_stats.N_alpha_beta[ordered_keys[key_index]]
    
    initial_prob_vector = count_vector / np.sum(count_vector) \
        if np.sum(count_vector) > 0 else count_vector
    markov_trans_matrix = np.zeros(count_matrix.shape)
    for i in range(count_matrix.shape[0]):
        row_sum = np.sum(count_matrix[i,:])
        if row_sum == 0:
            continue
        for j in range(count_matrix.shape[1]):
            markov_trans_matrix[i, j] = count_matrix[i, j] / row_sum

    return markov_trans_matrix, ordered_keys

def parse_transition_sequence(sequence_of_transitions, ordered_keys):
    """Parse raw existing_lines into a list of trajectory segments.
    
    Each segment is a list of milestone alias indices (ints) from 
    consecutive bounce events. Segments are split on "NEW_SWARM" entries.
    Only milestone indices present in ordered_keys are kept.
    
    Parameters
    ----------
    sequence_of_transitions : list
        The anchor_stats.existing_lines list. Each entry is either the 
        string "NEW_SWARM" or a list of 3 elements [dest_boundary, 
        bounce_index, time].
    ordered_keys : list of int
        Sorted milestone alias indices that define the matrix dimensions.
    
    Returns
    -------
    segments : list of list of int
        Each inner list is a trajectory segment of milestone alias indices.
    """
    if sequence_of_transitions is None:
        return []
    valid_keys = set(ordered_keys)
    segments = []
    current_segment = []
    for entry in sequence_of_transitions:
        if isinstance(entry, str) and "NEW_SWARM" in entry:
            if len(current_segment) > 0:
                segments.append(current_segment)
            current_segment = []
        else:
            # entry is [dest_boundary, bounce_index, time]
            milestone_idx = int(entry[0])
            if milestone_idx in valid_keys:
                current_segment.append(milestone_idx)
    # Don't forget the last segment
    if len(current_segment) > 0:
        segments.append(current_segment)
    return segments

def estimate_transition_matrix_at_lag(segments, ordered_keys, lag):
    """Estimate a transition matrix from trajectory segments at a given lag.
    
    For each segment, count transitions between the milestone observed at 
    position t and the milestone observed at position t + lag. Normalize 
    rows to get transition probabilities.
    
    Parameters
    ----------
    segments : list of list of int
        Trajectory segments from parse_transition_sequence.
    ordered_keys : list of int
        Sorted milestone alias indices.
    lag : int
        Lag in units of bounce events.
    
    Returns
    -------
    trans_matrix : numpy.ndarray or None
        Row-stochastic transition matrix of shape (M, M), or None if 
        insufficient data at this lag.
    total_transitions : int
        Total number of transition pairs counted.
    """
    M = len(ordered_keys)
    key_to_index = {key: i for i, key in enumerate(ordered_keys)}
    count_matrix = np.zeros((M, M))
    total_transitions = 0
    
    for segment in segments:
        if len(segment) <= lag:
            continue
        for t in range(len(segment) - lag):
            i = key_to_index.get(segment[t])
            j = key_to_index.get(segment[t + lag])
            if i is not None and j is not None:
                count_matrix[i, j] += 1
                total_transitions += 1
    
    if total_transitions == 0:
        return None, 0
    
    # Normalize rows
    trans_matrix = np.zeros((M, M))
    for i in range(M):
        row_sum = np.sum(count_matrix[i, :])
        if row_sum > 0:
            trans_matrix[i, :] = count_matrix[i, :] / row_sum
    
    return trans_matrix, total_transitions

def bootstrap_ck_test(segments, ordered_keys, lag_values, n_bootstrap=10):
    """Bootstrap error estimates for both predicted and estimated matrices.
    
    Resamples trajectory segments with replacement. For each resample,
    rebuilds the lag-1 matrix and all lag-n estimates from the same data.
    
    Returns
    -------
    boot_errors : dict
        Keys are lag values. Values are dicts with 'predicted_std' and
        'estimated_std', each an (M, M) array or None.
    """
    M = len(ordered_keys)
    # Collect bootstrap samples: {lag: {'predicted': [...], 'estimated': [...]}}
    samples = {lag: {'predicted': [], 'estimated': []} for lag in lag_values}
    rng = np.random.default_rng()

    resampled_segs = []
    for segment in segments:
        n_seg = len(segment)
        for _ in range(n_bootstrap):
            indices = rng.choice(n_seg, size=n_seg, replace=True)
            resampled = [segment[idx] for idx in indices]
            resampled_segs.append(resampled)
        
            # Build lag-1 matrix from resampled segments
            t1, _ = estimate_transition_matrix_at_lag(
                resampled_segs, ordered_keys, lag=1)
            if t1 is None:
                continue
            
            for lag in lag_values:
                predicted = np.linalg.matrix_power(t1, lag)
                estimated, _ = estimate_transition_matrix_at_lag(
                    resampled_segs, ordered_keys, lag)
                samples[lag]['predicted'].append(predicted)
                if estimated is not None:
                    samples[lag]['estimated'].append(estimated)
    
    boot_errors = {}
    for lag in lag_values:
        pred_list = samples[lag]['predicted']
        est_list = samples[lag]['estimated']
        boot_errors[lag] = {
            'predicted_std': np.std(pred_list, axis=0) 
                if len(pred_list) > 1 else np.zeros((M, M)),
            'estimated_std': np.std(est_list, axis=0) 
                if len(est_list) > 1 else None,
        }
        
    return boot_errors

def run_ck_test(anchor_transition_matrix, segments, ordered_keys, lag_values,
                n_bootstrap=10):
    """Run the Chapman-Kolmogorov test for a single anchor.
    
    Compares T(tau)^n (predicted) vs T(n*tau) (estimated from trajectory)
    for each lag n in lag_values. Optionally computes bootstrap errors.
    
    Parameters
    ----------
    anchor_transition_matrix : numpy.ndarray
        The lag-1 transition matrix of shape (M, M).
    segments : list of list of int
        Trajectory segments from parse_transition_sequence.
    ordered_keys : list of int
        Sorted milestone alias indices.
    lag_values : list of int
        Lag values to test (in units of bounce events).
    n_bootstrap : int
        Number of bootstrap resamples. Set to 0 to skip error estimation.
        Default is 10.
    Returns
    -------
    results : dict
        Keys are lag values. Values are dicts with:
        - 'predicted': T(1)^n (numpy array)
        - 'estimated': T(n) from trajectory (numpy array or None)
        - 'predicted_std': bootstrap std of predicted (numpy array or None)
        - 'estimated_std': bootstrap std of estimated (numpy array or None)
        - 'total_transitions': int count of transition pairs
        - 'frobenius_error': Frobenius norm of (predicted - estimated)
        - 'max_abs_error': max absolute element-wise deviation
    """
    results = {}
    for lag in lag_values:
        predicted = np.linalg.matrix_power(anchor_transition_matrix, lag)
        estimated, total_transitions = estimate_transition_matrix_at_lag(
            segments, ordered_keys, lag)
        
        result = {
            'predicted': predicted,
            'estimated': estimated,
            'total_transitions': total_transitions,
            'frobenius_error': None,
            'max_abs_error': None,
            'predicted_std': None,
            'estimated_std': None,
        }
        
        if estimated is not None:
            diff = predicted - estimated
            result['frobenius_error'] = np.linalg.norm(diff, 'fro')
            result['max_abs_error'] = np.max(np.abs(diff))
        
        results[lag] = result
    
    # Bootstrap error estimation
    if n_bootstrap > 0 and sum(len(seg) for seg in segments) > 1:
        boot_errors = bootstrap_ck_test(
            segments, ordered_keys, lag_values, n_bootstrap)
        for lag in lag_values:
            results[lag]['predicted_std'] = boot_errors[lag]['predicted_std']
            results[lag]['estimated_std'] = boot_errors[lag]['estimated_std']
    
    return results

def plot_ck_test(all_results, plot_dir, dpi=200):
    """Plot Chapman-Kolmogorov test results for all anchors.
    
    For each anchor, creates a subplot grid showing each matrix element
    T_ij(n) vs lag n, with predicted (line) and estimated (points).
    
    Parameters
    ----------
    all_results : dict
        Keys are anchor indices (int). Values are dicts with:
        - 'ordered_keys': list of milestone alias indices
        - 'ck_results': dict from run_ck_test
    plot_dir : str
        Directory to save plots.
    dpi : int
        Resolution of saved plots.
    """
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
    
    for alpha, anchor_data in all_results.items():
        ordered_keys = anchor_data['ordered_keys']
        ck_results = anchor_data['ck_results']
        M = len(ordered_keys)
        
        if M == 0 or len(ck_results) == 0:
            continue
        
        lag_values = sorted(ck_results.keys())
        
        fig, axes = plt.subplots(M, M, figsize=(3 * M, 3 * M), 
                                 squeeze=False)
        fig.suptitle(f"Chapman-Kolmogorov Test: Anchor {alpha}", 
                     fontsize=14)
        
        for i in range(M):
            for j in range(M):
                ax = axes[i][j]
                predicted_vals = []
                predicted_stds = []
                estimated_vals = []
                estimated_stds = []
                estimated_lags = []
                
                for lag in lag_values:
                    res = ck_results[lag]
                    predicted_vals.append(res['predicted'][i, j])
                    pred_std = res.get('predicted_std')
                    predicted_stds.append(
                        pred_std[i, j] if pred_std is not None else 0.0)
                    if res['estimated'] is not None:
                        estimated_vals.append(res['estimated'][i, j])
                        estimated_lags.append(lag)
                        est_std = res.get('estimated_std')
                        estimated_stds.append(
                            est_std[i, j] if est_std is not None else 0.0)
                
                predicted_vals = np.array(predicted_vals)
                predicted_stds = np.array(predicted_stds)

                ax.plot(lag_values, predicted_vals, '-', color='blue',
                        label='Predicted $T(1)^n$', linewidth=1.5)
                if np.any(predicted_stds > 0):
                    ax.fill_between(
                        lag_values,
                        predicted_vals - predicted_stds,
                        predicted_vals + predicted_stds,
                        alpha=0.25, color='blue')
                
                if len(estimated_vals) > 0:
                    ax.errorbar(estimated_lags, estimated_vals,
                                yerr=estimated_stds, fmt='o',
                                color='red', markersize=4, capsize=2,
                                label='Estimated $T(n)$')
                ax.set_title(f"$T_{{{ordered_keys[i]},{ordered_keys[j]}}}$",
                             fontsize=10)
                #ax.set_ylim(-0.05, 1.05)
                if i == M - 1:
                    ax.set_xlabel("Lag (bounces)")
                if j == 0:
                    ax.set_ylabel("Probability")
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
                ax.set_xscale('log')
        
        plt.tight_layout()
        
        if plot_dir is not None:
            plot_filename = os.path.join(
                plot_dir, f"ck_test_anchor_{alpha}.png")
            plt.savefig(plot_filename, dpi=dpi)
            print(f"CK test plot saved: {plot_filename}")
        
        plt.close(fig)

def chapman_kolmogorov_test(model, analysis, plot_dir=None, 
                            lag_values=None, dpi=200, n_bootstrap=10):
    """
    Perform a Chapman-Kolmogorov test for a SEEKR2 model and analysis.
    This will need to be done anchor by anchor.
    """
    if lag_values is None:
        lag_values = [1, 2, 5, 10, 20, 50]
    
    all_results = {}
    
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        anchor_stats = analysis.anchor_stats_list[alpha]
        # The trajectory information transition-by-transition.
        # existing_lines is a list, whose entries might be a string
        # containing "NEW_SWARM", or might be a list of length 3
        # where the first entry is the destination boundary alias index,
        # the second entry is a bounce index (count), and the third entry
        # is the time of the transition.
        sequence_of_transitions = anchor_stats.existing_lines
        result = make_anchor_transition_matrix(anchor, anchor_stats)
        if result is None:
            print(f"Anchor {alpha}: No transition data, skipping.")
            continue
        anchor_transition_matrix, ordered_keys = result
        
        # Parse the trajectory into segments
        segments = parse_transition_sequence(
            sequence_of_transitions, ordered_keys)
        
        if len(segments) == 0:
            print(f"Anchor {alpha}: No trajectory segments, skipping.")
            continue
        
        total_bounces = sum(len(seg) for seg in segments)
        print(f"Anchor {alpha}: {len(ordered_keys)} milestones, "
              f"{len(segments)} segments, {total_bounces} total bounces")
        
        # Run CK test
        ck_results = run_ck_test(
            anchor_transition_matrix, segments, ordered_keys, lag_values,
            n_bootstrap=n_bootstrap)
        
        all_results[alpha] = {
            'ordered_keys': ordered_keys,
            'ck_results': ck_results,
        }
        
        # Print summary
        for lag in lag_values:
            res = ck_results[lag]
            if res['frobenius_error'] is not None:
                print(f"  Lag {lag:4d}: Frobenius error = "
                      f"{res['frobenius_error']:.6f}, "
                      f"Max |error| = {res['max_abs_error']:.6f}, "
                      f"Transitions = {res['total_transitions']}")
            else:
                print(f"  Lag {lag:4d}: Insufficient data")
    
    # Plot results
    plot_ck_test(all_results, plot_dir, dpi=dpi)
    
    return all_results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-l", "--lags", dest="lags", default="1,2,5,10,20,50,100,200,500,1000",
        type=str, help="Comma-separated list of lag values (in units of "
        "bounce events) to test. Default: 1,2,5,10,20,50,100,200,500,1000")
    argparser.add_argument(
        "-d", "--dpi", dest="dpi", default=200, type=int,
        help="The DPI (dots per inch) of resolution for plots.")
    argparser.add_argument(
        "-n", "--n_bootstrap", dest="n_bootstrap", default=10, type=int,
        help="Number of bootstrap resamples for error estimation. "
        "Set to 0 to skip. Default: 10")
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    lag_values = [int(x.strip()) for x in args["lags"].split(",")]
    dpi = args["dpi"]
    n_bootstrap = args["n_bootstrap"]

    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, DEFAULT_IMAGE_DIR, CK_DIR)
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {plot_dir}")
    analysis = analyze.analyze(model, num_error_samples=0, skip_checks=True)
    chapman_kolmogorov_test(model, analysis, plot_dir, 
                            lag_values=lag_values, dpi=dpi,
                            n_bootstrap=n_bootstrap)
    