"""
Given a SEEKR project, with completed MMVT trajectories and MMVT output 
files, construct a "long" pseudo-trajectory out of the fragments, that
will mimic a long-timescale simulation exploring all the states along
the CV. This long trajectory will be used as input to AMINO and SGOOP
because it will have sampled the bound, unbound, and intermediate 
states.

Steps:
1. Load the SEEKR model. Open the MMVT output files, and find all the 
  "fragments". For each fragment in a given anchor, identify the "source" 
  milestone, the "destination" milestone, and the span of time the fragment
  existed.
    a. Need a Fragment() class
2. For each fragment, find the section of the .dcd file that describes
  that fragment's trajectory.
3. Begin constructing the long trajectory.
    a. Start in the "bound" anchor. Choose a fragment at random.
    b. Look at the fragment's destination milestone. That will be the next
      fragment's source milestone.
    c. The next fragment will be in the anchor on the opposite side of the 
      milestone. This fragment is chosen at random.
    d. Go to 3.b and repeat until all anchors have been sampled multiple
      times.
4. This will make a huge trajectory (milliseconds or seconds long, when
  each fragment may only be 100 fs or so). So we need a method to "cut"
  out extra frames to save space.
5. Another idea: Instead of making one long trajectory that samples all
  anchors, Have a series of not-so-long trajectories that go 2 or 3 
  anchor away.
    a. Once you have these, you would "stride" them - take out 9 out of 10 
      frames. Then this would be a new "Fragment" to piece together.
    b. We can use these larger Fragments to make the trajectories

Note: Use shorter trajectories made from fragments to find the local states
  between your anchors, and stitch your fragments to make more meaningful
  trajectories.
"""

import os
import glob
import math
import random
from collections import defaultdict
import time
import argparse

import mdtraj
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_base as mmvt_base

class Fragment():
    """
    A Fragment is a small trajectory moving within a single anchor,
    starting at one milestone and ending at a milestone, it may
    possibly end on the same milestone it came from.
    """
    def __init__(self, anchor_index, source_milestone, dest_milestone, 
                 start_time, end_time):
        self.anchor_index = anchor_index
        self.source_milestone = source_milestone
        self.dest_milestone = dest_milestone
        self.start_time = start_time
        self.end_time = end_time
        self.traj = None
        
        
    def extract_frames_from_dcd(self, model, traj):
        total_dcd_frames = model.calculation_settings.num_production_steps \
            // model.calculation_settings.trajectory_reporter_interval
        frames_per_picosecond = total_dcd_frames \
            / (model.calculation_settings.num_production_steps \
               * model.get_timestep())
        first_frame = int(math.ceil(self.start_time * frames_per_picosecond))
        last_frame = int(math.ceil(self.end_time * frames_per_picosecond))
        if first_frame == last_frame:
            self.traj = None
        else:
            self.traj = traj[first_frame:last_frame]
            
        return

def anchor_mmvt_output_slicer_dicer(model, anchor):
    output_file_glob = os.path.join(
        model.anchor_rootdir, anchor.directory, 
        anchor.production_directory, anchor.md_output_glob)
    output_file_list = glob.glob(output_file_glob)
    output_file_list = base.order_files_numerically(
        output_file_list)
    fragment_dict = defaultdict(list)
    src_milestone_alias = None
    start_time = None
    for output_file_name in output_file_list:
        with open(output_file_name) as f:
            for line in f.readlines():
                if line.startswith("#") or line.startswith("CHECKPOINT"):
                    continue
                line_split = line.strip().split(",")
                dest_milestone_alias = int(line_split[0])
                end_time = float(line_split[2])
                if src_milestone_alias is None:
                    src_milestone_alias = dest_milestone_alias
                    start_time = end_time
                    continue
                else:
                    source_milestone = anchor.id_from_alias(src_milestone_alias)
                    dest_milestone = anchor.id_from_alias(dest_milestone_alias)
                    fragment = Fragment(anchor.index, source_milestone, 
                                        dest_milestone, start_time, end_time)
                    fragment_dict[source_milestone].append(fragment)
                    src_milestone_alias = dest_milestone_alias
                    start_time = end_time
                    
                
    return fragment_dict

def load_anchor_dcd_files(model, anchor):
    building_directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory)
    prod_directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.production_directory)
    mmvt_traj_basename = mmvt_base.OPENMMVT_BASENAME+"*.dcd"
    mmvt_traj_glob = os.path.join(prod_directory, mmvt_traj_basename)
    mmvt_traj_filenames = glob.glob(mmvt_traj_glob)
    assert len(mmvt_traj_filenames) > 0, "Anchor {} has no dcd files.".format(
        anchor.index)
    if model.using_toy():
        pdb_filename = os.path.join(building_directory, "toy.pdb")
        top_filename = pdb_filename
    else:
        if anchor.amber_params.prmtop_filename is not None:
            prmtop_filename = os.path.join(
                building_directory, anchor.amber_params.prmtop_filename)
            top_filename = prmtop_filename
    
    traj = mdtraj.load(mmvt_traj_filenames, top=top_filename)
    return traj

def make_fragment_list(model):
    all_anchors_fragment_list = []
    for i, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        # Read the MMVT output files for this anchor
        anchor_fragment_dict = anchor_mmvt_output_slicer_dicer(model, anchor)
        traj = load_anchor_dcd_files(model, anchor)
        for key in anchor_fragment_dict:
            fragment_list = anchor_fragment_dict[key]
            for fragment in fragment_list:
                fragment.extract_frames_from_dcd(model, traj)
                
        all_anchors_fragment_list.append(anchor_fragment_dict)
    
    return all_anchors_fragment_list

def long_sequence_from_fragments(model, all_anchors_fragment_list, 
                         starting_anchor_index, max_time=None, 
                         min_visits_per_site=0, max_total_frames=1000):
    
    dcd_stride = model.get_timestep() \
        * model.calculation_settings.trajectory_reporter_interval
    
    if max_time is None:
        frame_stride = 1
    else:
        total_dcd_frames = int(max_time / dcd_stride)
        frame_stride = int(total_dcd_frames / max_total_frames)
        if frame_stride < 1:
            frame_stride = 1
    
    MAX_ITER = 100000000000
    current_anchor_index = starting_anchor_index
    
    anchor_frag_dict = all_anchors_fragment_list[starting_anchor_index]
    first_key = list(anchor_frag_dict.keys())[0]
    fragment_list = anchor_frag_dict[first_key]
    #current_fragment = random.choice(fragment_list)
    current_fragment_index = random.choice(range(len(fragment_list)))
    current_fragment = fragment_list[current_fragment_index]
    
    src_milestone = current_fragment.source_milestone
    
    incubation_time = current_fragment.end_time - current_fragment.start_time
    
    long_sequence = []
    
    counter = 0
    
    enough_time = False
    enough_states = False
    
    times_visited_states = []
    for anchor in model.anchors:
        if not anchor.bulkstate:
            times_visited_states.append(0)
    
    total_frame_counter = 0
    
    start_time = time.time()
    while True:
        if current_fragment.traj is not None:
            for traj_frame in range(current_fragment.traj.n_frames):
                total_frame_counter += 1
                if total_frame_counter % frame_stride == 0:
                    sequence_entry = [current_anchor_index, src_milestone, 
                                      current_fragment_index, traj_frame]
                    long_sequence.append(sequence_entry)

        current_anchor = model.anchors[current_anchor_index]
        dest_milestone_index = current_fragment.dest_milestone
        for milestone in current_anchor.milestones:
            if milestone.index == dest_milestone_index:
                dest_milestone = milestone
                break
        
        next_anchor_index = dest_milestone.neighbor_anchor_index
        if model.anchors[next_anchor_index].bulkstate:
            next_anchor_index = current_anchor_index
        
        if next_anchor_index != current_anchor_index:
            times_visited_states[current_anchor_index] += 1
        
        next_anchor_fragment_dict = all_anchors_fragment_list[next_anchor_index]
        next_anchor_fragment_list = next_anchor_fragment_dict[
            dest_milestone_index]
        assert len(next_anchor_fragment_list) > 0, \
            "No MMVT transitions found in anchor {} to milestone {}. "\
            .format(next_anchor_index, dest_milestone_index)\
            +"More simulation needed?"
        next_fragment_index = random.choice(range(len(
            next_anchor_fragment_list)))
        next_fragment = next_anchor_fragment_list[next_fragment_index]
        src_milestone = next_fragment.source_milestone       
        current_anchor_index = next_anchor_index
        current_fragment_index = next_fragment_index
        current_fragment = next_fragment
        incubation_time += current_fragment.end_time \
            - current_fragment.start_time
        
        if max_time is not None:
            if incubation_time > max_time:
                enough_time = True
                
        all_states_enough = True
        for times_visited_state in times_visited_states:
            if times_visited_state < min_visits_per_site:
                all_states_enough = False
        
        if all_states_enough:
            enough_states = True
        
        if enough_time and enough_states:
            break
        
        counter += 1
        if counter > MAX_ITER:
            raise Exception("Maximum iterations exceeded.")
    
    print("Time spent making sequence (s):", time.time() - start_time)
    
    return long_sequence

def trajectory_from_sequence(model, all_anchors_fragment_list, sequence, 
                             max_total_frames):
    long_trajectory = None
    while len(sequence) > max_total_frames:
        print("cutting sequence size from {} to {}.".format(len(sequence), 
                                                            len(sequence)//2))
        sequence = sequence[::2]
    
    for entry in sequence:
        [anchor_id, src_milestone, frag_index, traj_index] = entry
        fragment = all_anchors_fragment_list[anchor_id][src_milestone][frag_index]
        traj_frame = fragment.traj[traj_index]
        if long_trajectory is None:
            long_trajectory = traj_frame
        else:
            long_trajectory += traj_frame
            
    return long_trajectory

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Make a long trajectory "\
        "from piecing together the individual small fragmented trajectories "\
        "from the MMVT simulations.")
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="name of model file for SEEKR2 calculation. This would be the "\
        "XML file generated in the prepare stage.")
    argparser.add_argument(
        "output_dcd", metavar="OUTPUT_DCD", type=str, 
        help="Name of the file where the trajectory will be written.")
    argparser.add_argument(
        "-t", "--total_timespan", dest="total_timespan", 
        default=1000000.0, type=float,
        help="The amount of time that should the trajectory should span in "\
        "in units of picoseconds.")
    argparser.add_argument(
        "-f", "--max_total_frames", dest="max_total_frames", 
        default=1000, type=int,
        help="The maximum number of frames in the final DCD file.")
    argparser.add_argument(
        "-m", "--minimum_visits_per_anchor", dest="minimum_visits_per_anchor", 
        default=0, type=int,
        help="The minimum number of times every anchor should be visited "\
        "(that is, from another anchor).")
    argparser.add_argument(
        "-a", "--starting_anchor_index", dest="starting_anchor_index", 
        default=0, type=int,
        help="The index of the anchor to start the trajectory at.")
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    output_filename = args["output_dcd"]
    total_timespan_in_ps = args["total_timespan"]
    max_total_frames = args["max_total_frames"]
    minimum_visits_per_site = args["minimum_visits_per_anchor"]
    bound_anchor = args["starting_anchor_index"]
    
    model = base.load_model(model_file)
    starttime = time.time()
    all_anchors_fragment_list = make_fragment_list(model)
    print("Time to make fragment list (s):", time.time() - starttime)
    long_sequence = long_sequence_from_fragments(
        model, all_anchors_fragment_list, bound_anchor, total_timespan_in_ps, 
        minimum_visits_per_site, max_total_frames)
    long_trajectory = trajectory_from_sequence(
        model, all_anchors_fragment_list, long_sequence, max_total_frames)
    long_trajectory.save(output_filename)
