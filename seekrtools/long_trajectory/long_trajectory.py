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

import mdtraj
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_base as mmvt_base

class Fragment():
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
    assert len(mmvt_traj_filenames) > 0, "Anchor {} has no frames.".format(
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



def make_long_trajectory(model, all_anchors_fragment_list, starting_anchor_index, ITER = 200000):
    

    
    
    current_anchor_index = starting_anchor_index
    
    anchor_frag_dict = all_anchors_fragment_list[starting_anchor_index]
    first_key = list(anchor_frag_dict.keys())[0]
    fragment_list = anchor_frag_dict[first_key]
    current_fragment = random.choice(fragment_list)
    
    long_traj = current_fragment.traj[:]
    
    
    for i in range(ITER):
        current_anchor = model.anchors[current_anchor_index]
        dest_milestone_index = current_fragment.dest_milestone
        for milestone in current_anchor.milestones:
            if milestone.index == dest_milestone_index:
                dest_milestone = milestone
                break
        
        next_anchor_index = dest_milestone.neighbor_anchor_index
        try:
            next_anchor_fragment_dict = all_anchors_fragment_list[next_anchor_index]
        except IndexError:
            print("Trajectory terminated before final iteration at : " + str(i))
            break
        next_anchor_fragment_list = next_anchor_fragment_dict[dest_milestone.index]
        if len(next_anchor_fragment_list) == 0:
            print("Max Iteration number reached : " + str(i))
            break
        next_fragment = random.choice(next_anchor_fragment_list)
        long_traj += next_fragment.traj
        current_anchor_index = next_anchor_index
        current_fragment = next_fragment
   
    return long_traj
    

model_file = "/home/lvotapka/toy_seekr_systems/muller_potential/model.xml"

bound_anchor = 0

model = base.load_model(model_file)
        
all_anchors_fragment_list = make_fragment_list(model)

long_traj = make_long_trajectory(model, all_anchors_fragment_list, bound_anchor)

long_traj.save("muller_test.dcd")
