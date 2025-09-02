"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import os
import torch
import numpy as np
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import pihnn_devo.nn as nn_devo
import pihnn_devo.utils as utils_devo

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # set environment variable to avoid error message
)

out_dir = "../test/s03_test_cracked_plate_pihnn/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Network parameters
network_params = {
    "n_epochs": 6000,
    "learn_rate": 1e-5,
    "scheduler_apply": [500, 1000, 1500],
    "units": [1, 10, 10, 10, 1],
    "np_train": 300,
    "np_test": 20,
    "beta": 0.5,
    "gauss": 3,
}
utils_devo.export_network_params(params_dict=network_params, out_dir=out_dir)
# utils_devo.import_network_params(
#     filename="network_params.py", dir="../test/s01_collect_data_pihnn/"
# )


# -----------------------------------
# Geometry and boundary conditions
# -----------------------------------
h = 10  # 0.5*height of the domain
l = 10  # 0.5*length of the domain
n_segments = 50  # number of segments for each line

# import stress data and coordinates from text files
data_dir = "../test/s01_collect_data_pihnn/"
stress_file = data_dir + "stress_data.txt"
coords_file = data_dir + "coords_data.txt"
stress_data = np.loadtxt(stress_file, skiprows=1)  # Skip header row
coords_data = np.loadtxt(coords_file, skiprows=1)

stress_data = torch.Tensor(stress_data)
coords_data = torch.Tensor(coords_data)

# imposed constraints on top and bottom (traction/pure vertical compression)
sig_ext_t = 1j  # traction on top: σ_yy = +1
sig_ext_b = -1j  # traction on bottom: σ_yy = -1

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

# ----------------------------------------------------------
# horizontal crack in center of domain (-3,0) to (3,0)
# ----------------------------------------------------------
crack = geom.line(P1=-3 - 0j, P2=3 + 0j, bc_type=bc.stress_bc())
crack.add_crack_tip(tip_side=0)  # left crack tip
crack.add_crack_tip(tip_side=1)  # Right crack tip

boundary = geom.boundary(
    curves=[line1, line2, line3, line4, crack],
    np_train=network_params["np_train"],
    np_test=network_params["np_test"],
    enrichment="rice",
)

model = nn_devo.enriched_pihnn_devo(
    PDE="km", units=network_params["units"], boundary=boundary
)

model.initialize_weights(
    method="exp",
    beta=network_params["beta"],
    sample=boundary.extract_points(10 * network_params["np_train"])[0],
    gauss=network_params["gauss"],
)
loss_train, loss_test, _, _ = utils_devo.train_devo_adam(
    sig_xx_target=stress_data[:, 0],
    sig_yy_target=stress_data[:, 1],
    sig_xy_target=stress_data[:, 2],
    x_coords=coords_data[0],
    y_coords=coords_data[1],
    boundary=boundary,
    model=model,
    n_epochs=network_params["n_epochs"],
    learn_rate=network_params["learn_rate"],
    scheduler_apply=network_params["scheduler_apply"],
    scheduler_gamma=0.5,
    dir=out_dir,
)

tria = graphics.get_triangulation(boundary)
graphics.plot_sol(
    triangulation=tria,
    model=model,
    apply_crack_bounds=True,
    dir=out_dir,
)
graphics.plot_loss(
    loss_train=loss_train,
    loss_test=loss_test,
    dir=out_dir,
)
