"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import torch
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import pihnn_devo.nn as nn_devo
import pihnn_devo.utils as utils_devo

# Network parameters
# n_epochs = 6000  # Number of epochs
n_epochs = 10  # TODO: remove after debugging
learn_rate = 1e-4  # initial learning rate
scheduler_apply = [500, 1000, 1500]
units = [1, 10, 10, 10, 1]  # units in each network layer
np_train = 300  # number of training points on domain boundary
np_test = 20  # number of test points on the domain boundary
beta_param = 0.5  # initialization parameter
gauss_param = 3  # initialization parameter

# -----------------------------------
# Geometry and boundary conditions
# -----------------------------------
h = 10  # 0.5*height of the domain
l = 10  # 0.5*length of the domain
n_segments = 50  # number of segments for each line

# imposed constraints on top and bottom (traction/pure vertical compression)
sig_ext_t = 1j  # traction on top: σ_yy = +1
sig_ext_b = -1j

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

# ----------------------------------------------------------
# horizontal crack in center of domain (-3,0) to (3,0)
# ----------------------------------------------------------
crack = geom.line(P1=-3 - 0j, P2=3 + 0j, bc_type=bc.stress_bc())

sig_xx_target = torch.tensor(
    [-0.0853, 0.0067, 0.0068, -0.0859, -0.0860, 0.0076, 0.0087, -0.0836]
)
sig_yy_target = torch.tensor(
    [1.1788, 0.8001, 0.7978, 1.1751, 1.1767, 0.8000, 0.7986, 1.1748]
)
sig_xy_target = torch.tensor(
    [0.0538, 0.2477, -0.2496, -0.0526, -0.0532, -0.2485, 0.2490, 0.0531]
)

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

crack = geom.line(P1=-3 - 0j, P2=3 + 0j, bc_type=bc.stress_bc())
crack.add_crack_tip(tip_side=0)  # left crack tip
crack.add_crack_tip(tip_side=1)  # Eright crack tip

boundary = geom.boundary(
    [line1, line2, line3, line4, crack], np_train, np_test, enrichment="rice"
)

model = nn_devo.enriched_PIHNN_devo("km", units, boundary)

model.initialize_weights(
    method="exp",
    beta=beta_param,
    sample=boundary.extract_points(10 * np_train)[0],
    gauss=gauss_param,
)
loss_train, loss_test, ListeZ1 = utils_devo.train_devo_adam(
    sig_xx_target,
    sig_yy_target,
    sig_xy_target,
    boundary,
    model,
    n_epochs,
    learn_rate,
    scheduler_apply,
    scheduler_gamma=0.5,
    dir="../test/test_cracked_plate_phinn/",
)
graphics.plot_loss(
    loss_train=loss_train, loss_test=loss_test, dir="../test/test_cracked_plate_phinn/"
)
tria = graphics.get_triangulation(boundary)
graphics.plot_sol(
    triangulation=tria,
    model=model,
    apply_crack_bounds=True,
    dir="../test/test_cracked_plate_phinn/",
)
