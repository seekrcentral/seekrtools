import numpy as np
import matplotlib.pyplot as plt
import seekr2.modules.common_base as base

import plotting

title = "Muller Potential System"
model_file = "/home/lvotapka/toy_seekr_systems/muller_potential_cvs/model.xml"
model = base.load_model(model_file)
boundaries = np.array([[-1.5, 1.2], [-0.2, 2.0]])
toy_plot = plotting.Toy_plot(model, title, boundaries, stride=10)
milestone_cv_functions = ["0.75*x + value", "0.5*x + value", "0.65*x + value", "0.75*x + value", "1.0*x + value", "0.4*x + value", "0.0*x + value", "-0.25*x + value"]
plotting.draw_linear_milestones(toy_plot, milestone_cv_functions)
ani = toy_plot.animate_trajs(animating_anchor_indices=[0, 1, 2, 3, 4, 5, 6, 7])
plt.show()
