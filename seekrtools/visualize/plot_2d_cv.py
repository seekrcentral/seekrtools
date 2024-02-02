"""
A base module for plotting 2D CVs using the Voronoi tesselation anchor.
"""

import os
import re
import argparse
# DO NOT REMOVE: needed for toy plots
from math import exp

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import mdtraj
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

import seekr2.modules.common_base as base

PLOTS_DIRECTORY_NAME = "voronoi_tesselation_plots"
BACKGROUND_COLOR = "white" #"lightgrey"
TRAJ_POINT_RADIUS = 0.005

def refine_function_str(old_function_str):
        """
        Convert the energy landscape from a form used by OpenMM
        to a form used by Python.
        """
        new_function_str = re.sub("\^", "**", old_function_str)
        return new_function_str

def make_traj_cv_values(voronoi_cv, parm7_filename, dcd_filename):
    print(f"loading files: {parm7_filename} and {dcd_filename}")
    cv_vals = []
    for chunk in mdtraj.iterload(dcd_filename, top=parm7_filename, chunk=10):
        for frame_index in range(chunk.n_frames):
            sub_cv1 = voronoi_cv.child_cvs[0]
            sub_cv2 = voronoi_cv.child_cvs[1]
            radius1 = sub_cv1.get_mdtraj_cv_value(chunk, frame_index)
            radius2 = sub_cv2.get_mdtraj_cv_value(chunk, frame_index)
            values = [radius1, radius2]
            cv_vals.append(values)
                
    return cv_vals

def make_boundaries(anchor_values_list):
    margin = 0.2
    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9
    for anchor_value_pair in anchor_values_list:
        val_x = anchor_value_pair[0]
        val_y = anchor_value_pair[1]
        if val_x < min_x:
            min_x = val_x
        if val_x > max_x:
            max_x = val_x
        if val_y < min_y:
            min_y = val_y
        if val_y > max_y:
            max_y = val_y
    
    boundaries = [min_x - margin, max_x + margin, min_y - margin, 
                  max_y + margin]
    return boundaries

def my_draw_networkx_edge_labels(
        G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color="k",
        font_family="sans-serif", font_weight="normal", alpha=None, bbox=None,
        horizontalalignment="center", verticalalignment="center", ax=None,
        rotate=True, clip_on=True, rad=0):
    """
    TODO: replace with improved NetworkX edge labels, that are being 
    implemented in a subsequent release as of 01/05/2024
    
    Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=100,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def find_milestone_points(model, vor, boundaries):
    milestone_points = {}
    min_x = boundaries[0]
    max_x = boundaries[1]
    min_y = boundaries[2]
    max_y = boundaries[3]
    for i, ridge_point_pair in enumerate(vor.ridge_points):
        anchor1_index = ridge_point_pair[0]
        anchor2_index = ridge_point_pair[1]
        if (anchor1_index >= len(model.anchors)) \
                or (anchor2_index >= len(model.anchors)):
            continue
        
        anchor1 = model.anchors[anchor1_index]
        milestone_index = None
        for milestone in anchor1.milestones:
            if milestone.neighbor_anchor_index == anchor2_index:
                milestone_index = milestone.index
                break
            
        if milestone_index is None:
            continue
        
        ridge_vertex_pair = vor.ridge_vertices[i]
        for ridge_vertex_index in ridge_vertex_pair:
            vertex_inside_boundaries = True
            vertex = vor.vertices[ridge_vertex_index]
            if (vertex[0] < min_x) or (vertex[0] > max_x) \
                    or (vertex[1] < min_y) or (vertex[1] > max_y):
                vertex_inside_boundaries = False
                
            if (ridge_vertex_index == -1) or (not vertex_inside_boundaries):
                ridge_vertex_pair.remove(ridge_vertex_index)
        
        if len(ridge_vertex_pair) == 0:
            continue
        elif len(ridge_vertex_pair) == 2:
            vertex1_index = ridge_vertex_pair[0]
            vertex2_index = ridge_vertex_pair[1]
            vertex1 = vor.vertices[vertex1_index]
            vertex2 = vor.vertices[vertex2_index]
            avg_point_x = 0.5 * (vertex1[0] + vertex2[0])
            avg_point_y = 0.5 * (vertex1[1] + vertex2[1])
            avg_point = [avg_point_x, avg_point_y]
            milestone_points[milestone_index] = avg_point
        else:
            adj_point1 = vor.points[anchor1_index]
            adj_point2 = vor.points[anchor2_index]
            avg_point_x = 0.5 * (adj_point1[0] + adj_point2[0])
            avg_point_y = 0.5 * (adj_point1[1] + adj_point2[1])
            avg_point = [avg_point_x, avg_point_y]
            #vertex1_index = ridge_vertex_pair[0]
            #avg_point = vor.vertices[vertex1_index]
            milestone_points[milestone_index] = avg_point
        
    return milestone_points

def find_anchor_points(model):
    anchor_values = {}
    num_variables = len(model.anchors[0].variables)
    for alpha, anchor in enumerate(model.anchors):
        values = []
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            values.append(anchor.variables[var_name])
        anchor_values[alpha] = values
        
    return anchor_values

def plot_voronoi_tesselation(model, boundaries, fill_values=None, colorbar_label="Energy (kcal/mol)"):
    """
    Would work for either anchors or milestones
    """
    anchor_values = find_anchor_points(model)
    points = []
    num_variables = len(model.anchors[0].variables)
    for alpha, anchor in enumerate(model.anchors):
        #if anchor.bulkstate:
        #    continue
        if alpha in anchor_values:
            points.append(anchor_values[alpha])
        else:
            values = []
            for i in range(num_variables):
                var_name = "value_0_{}".format(i)
                values.append(anchor.variables[var_name])
    
    points.append([-100.0, -100.0])
    points.append([-100.0, 100.0])
    points.append([100.0, -100.0])
    points.append([100.0, 100.0])
    
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, line_width=2, show_vertices=False)
    
    ax = plt.gca()
    ax.axis(boundaries)
    ax.set_aspect('equal', adjustable='box')
    
    if fill_values is not None:
        fill_values_normalized = (fill_values-np.min(fill_values))\
            /(np.max(fill_values)-np.min(fill_values))
        
    for i in range(len(points)): #enumerate(vor.regions[:-5]):
        my_point_region = vor.point_region[i]
        region = vor.regions[my_point_region]
        if not -1 in region:
            polygon = [vor.vertices[j] for j in region]
            #if len(polygon) == 0: continue
            if fill_values is not None:
                jet = plt.cm.jet
                colormap = jet(fill_values_normalized[i])
            else:
                colormap = BACKGROUND_COLOR
            plt.fill(*zip(*polygon), color=colormap)
                
    if fill_values is not None:
        norm = matplotlib.colors.Normalize(np.min(fill_values), np.max(fill_values))
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=jet)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label(colorbar_label)
    
    return fig, ax, vor, points[:-4]

def plot_anchor_voronoi_tesselation(model, boundaries, anchor_fill_values=None,
                                    include_anchor_labels=True, colorbar_label="Energy (kcal/mol)"):
    
    fig, ax, vor, points = plot_voronoi_tesselation(
        model, boundaries, fill_values=anchor_fill_values, 
        colorbar_label=colorbar_label)
    
    if include_anchor_labels:
        for i, anchor in enumerate(model.anchors):
            index_str = f"{anchor.index}"
            point = points[i]
            plt.text(point[0], point[1], index_str, weight="bold")
    
    """ # Save for milestone plot
    milestone_points = find_milestone_points(model, vor, boundaries)
    
    for milestone_index, milestone_point in milestone_points.items():
        plt.text(milestone_point[0], milestone_point[1], str(milestone_index), 
                 color="lightgrey", fontsize="x-small")
    """
    
    return fig, ax, points

def plot_anchors(model, plot_dir, iteration, trajectory_values, boundaries, 
                 title, x_coordinate_title, y_coordinate_title, 
                 omit_iter_label=True, dpi=200, file_base="vt", 
                 draw_string=False, fill_values=None, edge_dictionary=None,
                 edge_labels=True, colorbar_label="Energy (kcal/mol)"):
    fig, ax, points = plot_anchor_voronoi_tesselation(model, boundaries, fill_values, colorbar_label=colorbar_label)
    min_x = boundaries[0]
    max_x = boundaries[1]
    min_y = boundaries[2]
    max_y = boundaries[3]
    if model.using_toy() and fill_values is None:
        my_color = "w"
        potential_energy_expression \
            = refine_function_str(
                model.toy_settings.potential_energy_expression)
        
        #boundaries = np.array([-1.5, 1.2, -0.2, 2.0])
        landscape_resolution=100
        max_Z=20
        bounds_x = np.linspace(min_x, max_x, landscape_resolution)
        bounds_y = np.linspace(min_y, max_y, landscape_resolution)
        X,Y = np.meshgrid(bounds_x, bounds_y)
        Z = np.zeros((landscape_resolution, landscape_resolution))
        for i, x1 in enumerate(bounds_x):
            for j, y1 in enumerate(bounds_y):
                # fill out landscape here
                Z[j,i] = eval(potential_energy_expression)
                if Z[j,i] > max_Z:
                    Z[j,i] = max_Z
        
        p = ax.pcolor(X, Y, Z, cmap=plt.cm.jet, vmin=Z.min(), vmax=Z.max())
        
        cbar = plt.colorbar(p)
        cbar.set_label("Energy (kcal/mol)")
    
    else:
        my_color = "k"
    
    ax.set_title(title)
    ax.set_xlabel(x_coordinate_title)
    ax.set_ylabel(y_coordinate_title)
    
    # Add trajectory points
    for alpha in trajectory_values:
        if model.anchors[alpha].bulkstate:
            continue
        for point in trajectory_values[alpha]:
            circle = plt.Circle(point, TRAJ_POINT_RADIUS, color=my_color, zorder=2.5, alpha=0.5)
            ax.add_patch(circle)
            
    # Add lines to make string
    if draw_string:
        points_array = np.array(points).T
        plt.plot(points_array[0], points_array[1], "o", linestyle="-", linewidth=2)
    
    # Add arrows showing connections between anchors
    if edge_dictionary is not None:
        pos = {}
        for i, point in enumerate(points):
            pos[i] = point
        
        edge_dict_nonzero = {}
        edge_label_values = {}
        for (key, value) in edge_dictionary.items():
            if value > 1e-9:
                edge_dict_nonzero[key] = value
                if isinstance(value, float):
                    edge_label_values[key] = f"{value:.1E}"
                else:
                    edge_label_values[key] = f"{value}"
        
        edge_list = []
        for key, value in edge_dict_nonzero.items():
            edge_list.append((key[0], key[1], {"w":f"{value}"}))
        
        G = nx.DiGraph(directed=True)
        G.add_edges_from(edge_list)
        #for key in edge_dictionary:
        #    G.add_edges_from([key])
        
        #pos=nx.spring_layout(G,seed=5)
        #nx.draw_networkx_nodes(G, pos, ax=ax)
        #nx.draw_networkx_labels(G, pos, ax=ax)
        
        #edgelist = [(u,v) for u,v in G.edges() if edge_dictionary[(u,v)] > 0]
        
        
        max_width = 3.0
        min_width = 0.3
        widths_dict = {}
        min_edge_value = min(edge_dict_nonzero.values())
        max_edge_value = max(edge_dict_nonzero.values())
        if min_edge_value == 0:
            max_log_edge_value = np.log(max_edge_value)
        else:
            max_log_edge_value = np.log(max_edge_value/min_edge_value)
        
        for key, value in edge_dict_nonzero.items():
            if (max_log_edge_value == 0) or (min_edge_value == 0):
                width = min_width
            else:
                width = min_width + (max_width - min_width) \
                    * (np.log(value / min_edge_value) \
                       / max_log_edge_value)
            widths_dict[key] = width
        
        edge_weights = []
        for edge in edge_list:
            key = (edge[0], edge[1])
            edge_weights.append(widths_dict[key])
        
        nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, edgelist=edge_list,
            connectionstyle="arc3,rad=0.5", width=edge_weights, node_size=0,
            min_source_margin=0, min_target_margin=0, edge_color="black")
         
        if edge_labels:
            my_draw_networkx_edge_labels(
                G, pos, ax=ax, edge_labels=edge_label_values, rotate=False, label_pos=0.7, 
                font_color="black", font_size=3, rad=0.5)
    
    if not omit_iter_label:
        # Add iteration label in upper corner
        time_str = "iteration: {}".format(iteration)
        font = {"weight":"bold"}
        plt.text(min_x+0.1, max_y-0.2, time_str, fontdict=font)
    
    if plot_dir is None:
        plt.show()
    else:
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        if iteration is None:
            plot_basename = f"{file_base}.png"
        else:
            plot_basename = f"{file_base}_{iteration}.png"
        plot_filename = os.path.join(plot_dir, plot_basename)
        print("Saving plot to:", plot_filename)
        plt.savefig(plot_filename, dpi=dpi)
        
    return

def plot_milestones(model, plot_dir, iteration, boundaries, 
                 title, x_coordinate_title, y_coordinate_title, 
                 omit_iter_label=True, dpi=200, file_base="vt", 
                 draw_string=False, fill_values=None, edge_dictionary=None,
                 edge_labels=True):
    NODE_SIZE = 80
    fig, ax, vor, anchor_points = plot_voronoi_tesselation(
        model, boundaries, fill_values=None)
    milestone_points = find_milestone_points(model, vor, boundaries)
    ax.set_title(title)
    ax.set_xlabel(x_coordinate_title)
    ax.set_ylabel(y_coordinate_title)
    
    pos = milestone_points
    G = nx.DiGraph(directed=True)
    G.add_nodes_from(list(range(model.num_milestones)))
    if fill_values is None:
        node_color = "grey"
    else:
        fill_values_normalized = (fill_values-np.min(fill_values))\
            /(np.max(fill_values)-np.min(fill_values))
        rainbow = plt.cm.rainbow
        node_color = []
        for i in range(model.num_milestones):
            node_color.append(rainbow(fill_values_normalized[i]))
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color=node_color)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4, font_color="black")
        
    if edge_dictionary is not None:
        
        edge_dict_nonzero = {}
        edge_label_values = {}
        for (key, value) in edge_dictionary.items():
            if value > 1e-9:
                edge_dict_nonzero[key] = value
                edge_label_values[key] = f"{value:.1E}"
                
        edge_list = []
        for key, value in edge_dict_nonzero.items():
            edge_list.append((key[0], key[1], {"w":f"{value}"}))
        
        G.add_edges_from(edge_list)
        #for key in edge_dictionary:
        #    G.add_edges_from([key])
        
        #pos=nx.spring_layout(G,seed=5)
        
        #edgelist = [(u,v) for u,v in G.edges() if edge_dict_nonzero[(u,v)] > 0]
        
        max_width = 3.0
        min_width = 0.3
        widths_dict = {}
        min_edge_value = min(edge_dict_nonzero.values())
        max_edge_value = max(edge_dict_nonzero.values())
        max_log_edge_value = np.log(max_edge_value/min_edge_value)
        for key, value in edge_dict_nonzero.items():
            if max_log_edge_value == 0:
                width = min_width
            else:
                width = min_width + (max_width - min_width) \
                    * (np.log(value / min_edge_value) \
                       / max_log_edge_value)
            widths_dict[key] = width
        
        edge_weights = []
        for edge in edge_list:
            key = (edge[0], edge[1])
            edge_weights.append(widths_dict[key])
        
        nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, edgelist=edge_list, node_size=0.5*NODE_SIZE,
            connectionstyle="arc3,rad=0.5", width=edge_weights, 
            min_source_margin=0, min_target_margin=0, edge_color="black")
         
        if edge_labels:
            my_draw_networkx_edge_labels(
                G, pos, ax=ax, edge_labels=edge_label_values, rotate=False, label_pos=0.7, 
                font_color="black", font_size=3, rad=0.5)

    if plot_dir is None:
        plt.show()
    else:
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        if iteration is None:
            plot_basename = f"{file_base}.png"
        else:
            plot_basename = f"{file_base}_{iteration}.png"
        plot_filename = os.path.join(plot_dir, plot_basename)
        print("Saving plot to:", plot_filename)
        plt.savefig(plot_filename, dpi=dpi)
        
    return

def make_model_plots(model, plot_dir=None,
                    x_coordinate_title="X-Coordinate", 
                    y_coordinate_title="Y-Coordinate", omit_iter_label=False, 
                    dpi=100, boundaries=None, traj_values=None, base_name="vt",
                    draw_string=False, anchor_fill_values=None, 
                    milestone_fill_values=None):
    anchor_values = find_anchor_points(model)
    if boundaries is None:
        #point_values = [anchor_values]
        point_values = list(anchor_values.values())
        if traj_values is None:
            traj_values = []
        else:
            point_values += traj_values[0]
        boundaries = make_boundaries(point_values)
    plot_anchors(
        model, plot_dir, None, traj_values, boundaries, 
        "Anchor Voronoi Tesselation", x_coordinate_title, y_coordinate_title, omit_iter_label, dpi, 
        base_name, draw_string, fill_values=anchor_fill_values)
    
    edge_dictionary = {}
    for anchor in model.anchors:
        for milestone in anchor.milestones:
            key = (anchor.index, milestone.neighbor_anchor_index)
            edge_dictionary[key] = 1.0
            
    plot_anchors(
        model, plot_dir, None, traj_values, boundaries, 
        "Connections between Anchors", x_coordinate_title, y_coordinate_title, omit_iter_label, dpi, 
        base_name+"_edges", draw_string, fill_values=anchor_fill_values, 
        edge_dictionary=edge_dictionary, edge_labels=False)
    
    plot_milestones(model, plot_dir, None, boundaries, 
                 "Milestones", x_coordinate_title, y_coordinate_title, 
                 omit_iter_label=True, dpi=200, file_base=base_name+"_milestones", 
                 draw_string=False, fill_values=None, edge_dictionary=None,
                 edge_labels=True)
    
    
    
    return boundaries

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-T", "--topology_file", dest="topology_file", default=None, type=str, 
        help="The name of topology file for visualizing trajectory points.")
    argparser.add_argument(
        "-D", "--traj_file", dest="traj_file", default=None, type=str, 
        help="The name of trajectory file for visualizing trajectory points. "\
        "PDB trajectories not recommended: use DCD or other format.")
    #argparser.add_argument(
    #    "-t", "--title", dest="title", default="Voronoi Tesselation",
    #    type=str, help="The title of the plot")
    argparser.add_argument(
        "-x", "--x_coordinate_title", dest="x_coordinate_title", 
        default="X-Coordinate",
        type=str, help="The title of x-coordinate")
    argparser.add_argument(
        "-y", "--y_coordinate_title", dest="y_coordinate_title", 
        default="Y-Coordinate",
        type=str, help="The title of y-coordinate")
    argparser.add_argument(
        "-d", "--dpi", dest="dpi", default=200, type=int,
        help="The DPI (dots per inch) of resolution for plots.")
        
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    topology_file = args["topology_file"]
    traj_file = args["traj_file"]
    #title = args["title"]
    x_coordinate_title = args["x_coordinate_title"]
    y_coordinate_title = args["y_coordinate_title"]
    dpi = args["dpi"]
    
    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    
    if topology_file is not None:
        assert traj_file is not None, \
            "Both topology_file and traj_file must be defined, or neither."
        voronoi_cv = model.collective_variables[0]
        cv_values = make_traj_cv_values(voronoi_cv, topology_file, traj_file)
        traj_values = {0:cv_values}
    else:
        assert traj_file is None, \
            "Both topology_file and traj_file must be defined, or neither."
        traj_values = None
    
    make_model_plots(model, plot_dir, x_coordinate_title, 
                     y_coordinate_title, True, dpi, traj_values=traj_values)
    
