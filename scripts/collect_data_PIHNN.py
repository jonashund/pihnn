"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import torch
import pihnn.nn as nn
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import pihnn_devo.utils as utils_devo

# Network parameters
n_epochs = 10000  # Number of epochs
learn_rate = 1e-3  # Initial learning rate
scheduler_apply = [2000, 4500, 7000, 8000, 9500]
units = [1, 10, 10, 10, 1]  # Units in each network layer
np_train = 250  # Number of training points on domain boundary
np_test = 20  # Number of test points on the domain boundary
beta = 0.5  # Initialization parameter
gauss = 3  # Initialization parameter


h = 10  # Half-height of the domain
l = 10  # Half-length of the domain

# Applied stresses at top and bottom (pure vertical tension/compression)
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
    np_train=np_train,
    np_test=np_test,
    enrichment="rice",
)

# Definition of NN
model = nn.enriched_PIHNN("km", units, boundary)

if __name__ == "__main__":
    model.initialize_weights(
        "exp", beta, boundary.extract_points(10 * np_train)[0], gauss
    )
    loss_train, loss_test = utils_devo.train(
        boundary, model, n_epochs, learn_rate, scheduler_apply
    )
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(
        tria, model, apply_crack_bounds=True
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

print("sig_xx_target : ", sig_xx_target)
print("sig_yy_target : ", sig_yy_target)
print("sig_xy_target : ", sig_xy_target)
