"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import os
import torch
import numpy as np
import pihnn.nn as nn
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import pihnn.utils as utils
import pihnn_devo.utils as utils_devo

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # set environment variable to avoid error message
)

out_dir = "../test/s01_collect_data_pihnn/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

network_params = {
    "n_epochs": 5000,
    "learn_rate": 1e-3,
    "scheduler_apply": [2000, 4500],
    "units": [1, 10, 10, 10, 1],
    "np_train": 250,
    "np_test": 20,
    "beta": 0.5,
    "gauss": 3,
}
utils_devo.export_network_params(params_dict=network_params, out_dir=out_dir)

h = 10  # half-height of the domain
l = 10  # half-length of the domain

# applied stresses at top and bottom (pure vertical tension/compression)
sig_ext_t = 1j  # Top tension: σ_yy = +1
sig_ext_b = -1j

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
# line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.displacement_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

z1 = -3.5 - 3.5j
z2 = 3.5 + 3.5j

z1N = z1 / (l)
z2N = z2 / (l)

crack = geom.line(P1=z1N, P2=z2N, bc_type=bc.stress_bc())

# Definition of crack tips
crack.add_crack_tip(tip_side=0)  # Left tip
crack.add_crack_tip(tip_side=1)  # Right tip

# Construction of the complete boundary with Rice-type XFEM enrichment
boundary = geom.boundary(
    curves=[line1, line2, line3, line4, crack],
    np_train=network_params["np_train"],
    np_test=network_params["np_test"],
    enrichment="rice",
)

# Definition of NN
model = nn.enriched_PIHNN("km", network_params["units"], boundary)

model.initialize_weights(
    "exp",
    network_params["beta"],
    boundary.extract_points(10 * network_params["np_train"])[0],
    network_params["gauss"],
)
loss_train, loss_test = utils.train(
    boundary=boundary,
    model=model,
    n_epochs=network_params["n_epochs"],
    learn_rate=network_params["learn_rate"],
    scheduler_apply=network_params["scheduler_apply"],
    dir=out_dir,
)
graphics.plot_loss(loss_train, loss_test, dir=out_dir)
tria = graphics.get_triangulation(boundary)
graphics.plot_sol(
    tria, model, apply_crack_bounds=True, dir=out_dir
)  # We bound the crack singularities for the plot

x_vals = torch.tensor([-6.0, -3.0, 3.0, 6.0])
y_vals = torch.tensor([6.0, 3.0, -3.0, -6.0])  # added -3.0 and 3.0

# Normalization
x_vals_norm = x_vals / l
y_vals_norm = y_vals / l

# Construction of z_data with normalized values
z_data = torch.cat([x_vals_norm + 1j * y for y in y_vals_norm])
z_data = z_data.to(torch.cfloat).requires_grad_(True)

# Expected values (e.g. zero stress at free points)
sig_xx_target, sig_yy_target, sig_xy_target, _, _ = model(z_data, real_output=True)

# export data to text file
data = torch.stack((sig_xx_target, sig_yy_target, sig_xy_target), dim=1)
data_np = data.detach().numpy()
output_file = out_dir + "stress_data.txt"
np.savetxt(output_file, data_np, header="sig_xx, sig_yy, sig_xy")

x_coords = (z_data.real).detach().numpy()
y_coords = (z_data.imag).detach().numpy()
coords_file = out_dir + "coords_data.txt"
np.savetxt(coords_file, [x_coords, y_coords], header="x, y")
