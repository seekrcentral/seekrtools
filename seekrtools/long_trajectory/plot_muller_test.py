import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seekr2.modules.common_base as base

import plotting
potential_energy_expression = "-20*exp(-1 * (x1 - 1)^2 + 0 * (x1 - 1) * (y1 - 0) - 10 * (y1 - 0)^2) - 10*exp(-1 * (x1 - 0)^2 + 0 * (x1 - 0) * (y1 - 0.5) - 10 * (y1 - 0.5)^2) - 17*exp(-6.5 * (x1 + 0.5)^2 + 11 * (x1 + 0.5) * (y1 - 1.5) - 6.5 * (y1 - 1.5)^2) + 1.5*exp(0.7 * (x1 + 1)^2 + 0.6 * (x1 + 1) * (y1 - 1) + 0.7 * (y1 - 1)^2)"
title = "Muller Potential System"
dcd_path = "muller_test.dcd"
topology = "/home/lvotapka/toy_seekr_systems/muller_potential/anchor_0/building/toy.pdb"
boundaries = np.array([[-1.5, 1.2], [-0.2, 2.0]])
#milestone_function = "0.66*x + value"
toy_plot = plotting.Toy_plot(potential_energy_expression, title, boundaries, 
    stride=100, timestep=0.002, steps_per_frame=10)
#plotting.draw_linear_milestones(milestone_function)

ani = toy_plot.animate_trajs(dcd_path, topology=topology)
plt.show()

#movie_filename = "muller_potential_long.gif" 
#writergif = animation.ImageMagickWriter(fps=30) 
#ani.save(movie_filename, writer=writergif)

#movie_filename = "muller_potential_long.mp4" 
#writervideo = animation.FFMpegWriter(fps=60) 
#ani.save(movie_filename, writer=writervideo)