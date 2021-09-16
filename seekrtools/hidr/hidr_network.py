"""
Utilities for exploring the anchor network within a model in order to
find the optimal procedure for steering the system to the different
anchors.
"""

import numpy as np
try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit

def find_edge_distance(model, src_anchor_index, dest_anchor_index):
    """
    Find the distance of the variables commonly shared between two
    anchors. The distance is assumed to be Euclidean.
    
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors.
    src_anchor_index : integer
        The index of the anchor within the model that will be the
        'source' - where the steered simulation will start from.
    dest_anchor_index : integer
        The index of the anchor within the model that will be the
        'destination' - where the steered simulation will go to.
        
    Returns:
    --------
    distance : float
        The distance (in nm) between the source and destination anchors.
    """
    # TODO: this may need to eventually be adapted to angular CVs
    variable_distance_sq = 0.0
    for variable in model.anchors[src_anchor_index].variables:
        if variable in model.anchors[dest_anchor_index].variables:
            distance = model.anchors[src_anchor_index].variables[variable] \
                - model.anchors[dest_anchor_index].variables[variable]
            variable_distance_sq += distance**2
            
    return np.sqrt(variable_distance_sq)

def find_next_anchor_index(model, visited_anchor_dict):
    """
    Given a set of explored anchors, find a candidate for the next step
    in the pathfinding based on which will be the closest anchor (node)
    to reach.
    
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors.
    visited_anchor_dict : dict
        A dictionary whose keys are anchor indices and whose values are
        a distance to a starting anchor.
        
    Returns:
    --------
    prev_anchor_index : int
        For the next step, starting at this anchor will be the step that
        will explore the closest available node (anchor).
    next_anchor_index : int
        For the next step, this anchor will be the step that explores 
        the closest available node (anchor).
    next_anchor_distance : float
        The total distance from one of the starting nodes (anchors) to
        the destination of this step.
    """
    prev_anchor_index = None
    next_anchor_index = None
    next_anchor_distance = 9e9
    for visited_anchor_index in visited_anchor_dict:
        for milestone in model.anchors[visited_anchor_index].milestones:
            neighbor_index = milestone.neighbor_anchor_index
            if model.anchors[neighbor_index].bulkstate:
                continue
            if neighbor_index not in visited_anchor_dict:
                edge_distance = find_edge_distance(model, visited_anchor_index, 
                                               neighbor_index)
                total_distance = visited_anchor_dict[visited_anchor_index] \
                    + edge_distance
                if total_distance < next_anchor_distance:
                    next_anchor_distance = total_distance
                    prev_anchor_index = visited_anchor_index
                    next_anchor_index = neighbor_index
    
    return prev_anchor_index, next_anchor_index, next_anchor_distance
        

def get_procedure(model, source_anchor_indices, destination_list):
    """
    Given the model and a list of starting anchors, a procedure
    will be provided that will allow every anchor to be reached
    from one of the starting anchors.
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors.
    source_anchor_indices : int
        Indices of anchors which have starting structures defined.
        The network will be explored starting from these nodes.
    destination_list : list
        A list of anchor indices that need to be reached by this
        procedure.
        
    Returns:
    --------
    procedure : list
        A list of tuples of integers, where each tuple is a step in a
        procedure of an SMD simulation from a source anchor to a
        destination anchor.
    """
    procedure = []
    visited_anchor_dict = {}
    for source_anchor_index in source_anchor_indices:
        visited_anchor_dict[source_anchor_index] = 0.0
    
    for i in range(len(model.anchors)):
        prev_anchor_index, next_anchor_index, next_anchor_distance \
            = find_next_anchor_index(model, visited_anchor_dict)
        if next_anchor_index is None:
            # No more paths found
            break
        visited_anchor_dict[next_anchor_index] = next_anchor_distance
        procedure.append((prev_anchor_index, next_anchor_index))
        if next_anchor_index in destination_list:
            destination_list.remove(next_anchor_index)
        if len(destination_list) == 0:
            break
        
    return procedure

def estimate_simulation_time(model, procedure, velocity):
    """
    Estimate the total amount of time will be needed to simulate enough
    SMD to fill all empty anchors.
    
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors.
    procedure : list
        The output from get_procedure - a list of tuples of anchor
        indices indicating the steps to take to reach each anchor with
        a series of SMD simulations.
    velocity : Quantity
        The speed (in velocity units) that the system will be moving 
        between anchors due to guided restraints.
        
    Returns:
    --------
    total_time : Quantity
        The total time estimate that would be required to complete the
        provided procedure.
    """
    total_time = 0.0 * unit.nanoseconds
    for step in procedure:
        source_anchor_index = step[0]
        destination_anchor_index = step[1]
        edge_distance = find_edge_distance(
            model, source_anchor_index, destination_anchor_index) \
            * unit.nanometers
        time_in_step = edge_distance / velocity
        total_time += time_in_step
        
    return total_time