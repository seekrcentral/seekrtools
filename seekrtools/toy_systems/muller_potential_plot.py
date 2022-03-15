import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seekr2.modules.common_base as base

import plotting

title = "Muller Potential System"
model_file = "/home/lvotapka/toy_seekr_systems/muller_potential/model.xml"
model = base.load_model(model_file)
boundaries = np.array([[-1.5, 1.2], [-0.2, 2.0]])
toy_plot = plotting.Toy_plot(model, title, boundaries, stride=25)
milestone_cv_functions = ["0.66*x + value"]
plotting.draw_linear_milestones(toy_plot, milestone_cv_functions)
ani = toy_plot.animate_trajs(animating_anchor_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
#plt.show()

#movie_filename = "muller_potential_milestones.gif" 
#writergif = animation.ImageMagickWriter(fps=30) 
#ani.save(movie_filename, writer=writergif)

movie_filename = "muller_potential_milestones.mp4" 
writervideo = animation.FFMpegWriter(fps=60) 
ani.save(movie_filename, writer=writervideo)
