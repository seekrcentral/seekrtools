"""
committor.py

Compute and plot committor probabilities for a SEEKR2 calculation.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seekr2.modules.common_base as seekr2_base
import seekr2.modules.common_analyze as seekr2_common_analyze
import seekr2.analyze as analyze

DEFAULT_NUM_ERROR_SAMPLES = 100

def committor(model):
    """
    Return the committor probabilities for a set of
    SEEKR2 milestones.
    """
    N = model.num_milestones
    committor_probabilities = np.zeros(N)
    committor_probabilities_error = np.zeros(N)
    analysis = analyze.analyze(model, skip_checks=True, num_error_samples=DEFAULT_NUM_ERROR_SAMPLES)
    assert analysis.main_data_sample.K_hat is not None
    K_hat = analysis.main_data_sample.K_hat
    assert N == K_hat.shape[0]
    source_vec = np.zeros((N,1))
    K_hat_inf = np.linalg.matrix_power(K_hat, seekr2_common_analyze.MATRIX_EXPONENTIAL)
    for i in range(N):
        source_vec[i,0] = 1.0
        final_probs = np.dot(K_hat_inf.T, source_vec)
        committor_probabilities[i] = final_probs[0,0]
        source_vec[i,0] = 0.0
    
    # Now let's assign errors
    for err in range(DEFAULT_NUM_ERROR_SAMPLES):
        K_hat = analysis.data_sample_list[err].K_hat
        assert N == K_hat.shape[0]
        source_vec = np.zeros((N,1))
        K_hat_inf = np.linalg.matrix_power(K_hat, seekr2_common_analyze.MATRIX_EXPONENTIAL)
        for i in range(N):
            source_vec[i,0] = 1.0
            final_probs = np.dot(K_hat_inf.T, source_vec)
            committor_probabilities_error[i] += (final_probs[0,0] - committor_probabilities[i])**2
            source_vec[i,0] = 0.0

    committor_probabilities_error = np.sqrt(committor_probabilities_error / DEFAULT_NUM_ERROR_SAMPLES)
    return committor_probabilities, committor_probabilities_error

def plot_committor(model, committor_probabilities, committor_probabilities_error,
                   image_directory):
    """
    Plot the committor probabilities for a set of
    SEEKR2 milestones.
    """
    pi_fig, ax = plt.subplots()
    plt.errorbar(
        range(model.num_milestones), committor_probabilities,
        yerr=committor_probabilities_error, ecolor="k", capsize=2)
    plt.xticks(range(model.num_milestones), range(model.num_milestones), rotation=90)
    plt.ylabel("Committor probability")
    plt.xlabel("Milestone index")
    #plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    pi_fig.savefig(os.path.join(
        image_directory, "committor_by_milestone.png"))
    return

def find_milestone_closest_to_halfway(model, committor_probabilities):
    """
    Return the index of the milestone with committor probability closest to 0.5.
    """
    return np.argmin(np.abs(committor_probabilities - 0.5))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    model = seekr2_base.load_model(model_file)
    image_directory = seekr2_common_analyze.make_image_directory(
        model, seekr2_common_analyze.DEFAULT_IMAGE_DIR)
    committor_probabilities, committor_probabilities_error = committor(model)
    closest_to_halfway_index = find_milestone_closest_to_halfway(model, committor_probabilities)
    #print("Committor probabilities by milestone: ", committor_probabilities)
    #print("Committor probability errors by milestone: ", committor_probabilities_error)
    print("Milestone index closest to halfway: ", closest_to_halfway_index)
    plot_committor(model, committor_probabilities, committor_probabilities_error,
                   image_directory)
    