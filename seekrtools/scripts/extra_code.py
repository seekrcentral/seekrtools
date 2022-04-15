"""
Extra code goes here that might come in handy someday.

"""

def make_milestoning_objects_nd(self, milestone_alias, milestone_index, 
                                 indices, which_index, widths_list):
        milestones = []
        index = indices[which_index]
        i = which_index
        if index > 0:
            neighbor_indices = indices[:]
            neighbor_indices[i] -= 1
            neighbor_index = get_anchor_index(neighbor_indices, widths_list)
            milestone1 = base.Milestone()
            milestone1.index = milestone_index
            milestone1.neighbor_index = neighbor_index
            milestone1.alias_id = milestone_alias
            milestone1.cv_index = self.index
            milestone1.variables = {"k": -1.0, "radius": self.radii[index-1]}
            """
            print("creating milestone: index:", milestone1.index,
                  "neighbor_index:", milestone1.neighbor_index,
                  "alias_id:", milestone1.alias_id, "cv_index:", 
                  milestone1.cv_index, "variables:", milestone1.variables)
            """
            milestone_alias += 1
            milestone_index += 1
            milestones.append(milestone1)
        
        if index < widths_list[i] - 1:
            neighbor_indices = indices[:]
            neighbor_indices[i] += 1
            neighbor_index = get_anchor_index(neighbor_indices, widths_list)
            milestone2 = base.Milestone()
            milestone2.index = milestone_index
            milestone2.neighbor_index = neighbor_index
            milestone2.alias_id = milestone_alias
            milestone2.cv_index = self.index
            milestone2.variables = {"k": 1.0, "radius": self.radii[index]}
            """
            print("creating milestone: index:", milestone2.index,
                  "neighbor_index:", milestone2.neighbor_index,
                  "alias_id:", milestone2.alias_id, "cv_index:", 
                  milestone2.cv_index, "variables:", milestone2.variables)
            """
            milestone_alias += 1
            milestone_index += 1
            milestones.append(milestone2)
        
        return milestones, milestone_alias, milestone_index
        
        
def create_grid_distance_cv_anchors_and_milestones(
        widths_list, cv_indices, cv_inputs, _dimension_index=0, _indices=None, 
        _milestone_index=0):
    """
    Create a set of Anchors and Milestones which conform to a 'grid'
    of milestones embedded in N (probably orthogonal) dimensions.
    """
    num_dimensions = len(widths_list)
    if _indices is None:
        _indices = []
        for i in range(num_dimensions):
            _indices.append(0)            
            
    if _dimension_index == num_dimensions:
        # base case
        
        anchor = base.Anchor()
        anchor.index = get_anchor_index(_indices, widths_list)
        anchor.name = "anchor_"+"_".join(str(index) for index in _indices)
        anchor.directory = anchor.name
        anchor.md = True
        #print("creating anchor named:", anchor.name, "index:", anchor.index)
        milestone_alias = 1
        for i, index in enumerate(_indices):
            # TODO: obtain milestone object from CV object
            cv_input = cv_inputs[i]
            milestones, milestone_alias, _milestone_index = \
                cv_input.make_milestoning_objects(
                    milestone_alias, _milestone_index, 
                    _indices, i, widths_list)
            anchor.milestones += milestones
        
        return _milestone_index, [anchor]
        
    else:
        anchor_list = []
        for i in range(widths_list[_dimension_index]):
            _indices[_dimension_index] = i
            _milestone_index, new_anchors = \
                create_grid_distance_cv_anchors_and_milestones(
                widths_list, cv_indices, cv_inputs, 
                _dimension_index=_dimension_index+1, _indices=_indices, 
                _milestone_index=_milestone_index)
            anchor_list += new_anchors
        
        return _milestone_index, anchor_list
            


def get_anchor_index(dimension_indices, width_list):
    """
    
    """
    num_dimensions = len(dimension_indices)
    anchor_index = 0
    total_width = 1
    for i in range(num_dimensions):
        j = (num_dimensions - 1) - i
        anchor_index += total_width * dimension_indices[j]
        total_width *= width_list[j]
    return anchor_index

        
        
    # Old code that could create a grid of anchors with N CVs   
    #_, anchors = create_grid_distance_cv_anchors_and_milestones(
    #    widths_list, cv_indices, collective_variable_inputs)

