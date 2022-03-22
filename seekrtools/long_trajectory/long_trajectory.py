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

import seekr2.modules.common_base as base

class Fragment():
    def __init__(self, anchor_index, source_milestone, dest_milestone, 
                 start_time, end_time):
        self.anchor_index = anchor_index
        self.source_milestone = source_milestone
        self.dest_milestone = dest_milestone
        self.start_time = start_time
        self.end_time = end_time
        
    def extract_frames_from_dcd(self, dcd_file):
        pass

def anchor_mmvt_output_slicer_dicer(model, anchor):
    output_file_glob = os.path.join(
        model.anchor_rootdir, anchor.directory, 
        anchor.production_directory, anchor.md_output_glob)
    output_file_list = glob.glob(output_file_glob)
    output_file_list = base.order_files_numerically(
        output_file_list)
    fragment_list = []
    src_milestone_alias = None
    start_time = None
    for output_file_name in output_file_list:
        with open(output_file_name) as f:
            for line in f.readlines():
                if line.startswith("#"):
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
                    fragment_list.append(fragment)
                    src_milestone_alias = dest_milestone_alias
                    start_time = end_time
                    
    return fragment_list
          
def make_fragment_list(model):
    all_anchors_fragment_list = []
    for i, anchor in enumerate(model.anchors):
        # Read the MMVT output files for this anchor
        anchor_fragment_list = anchor_mmvt_output_slicer_dicer(model, anchor)
        all_anchors_fragment_list.append(anchor_fragment_list)
    
    return

model_file = "/home/lvotapka/toy_seekr_systems/muller_potential/model.xml"

model = base.load_model(model_file)
        
make_fragment_list(model)

    

