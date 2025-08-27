"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import sys

# sys.path.append(r"C:\Windows\System32\site-packages\python3.10")
sys.path.append(r"C:\Users\Nicolas\Desktop\Stage_2A\Deep Learning\pihnn-main")

import pihnn.crack_detection as cd
import os
import torch
import scipy
import numpy as np
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import pihnn.crack_finding as cf

# Network parameters
n_epochs = 6000  # Number of epochs
learn_rate = 1e-4  # Initial learning rate
scheduler_apply = [1000, 2000, 3200, 4500, 5500]
units = [1, 10, 10, 10, 1]  # Units in each network layer
np_train = 300  # Number of training points on domain boundary
np_test = 20  # Number of test points on the domain boundary
beta = 0.5  # Initialization parameter
gauss = 3  # Initialization parameter

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
# horizontal crack in center of domain (-3 to 3)
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

n_epochs = 2000  # Number of epochs
learn_rate = 1e-5  # Initial learning rate
scheduler_apply = [500, 1000, 1500]

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

crack = geom.line(P1=-3 - 0j, P2=3 + 0j, bc_type=bc.stress_bc())
crack.add_crack_tip(tip_side=0)  # Extrémité gauche
crack.add_crack_tip(tip_side=1)  # Extrémité droite

boundary = geom.boundary(
    [line1, line2, line3, line4, crack], np_train, np_test, enrichment="rice"
)

model = nn.enriched_PIHNN_finding("km", units, boundary)
if __name__ == "__main__":
    model.initialize_weights(
        "exp", beta, boundary.extract_points(10 * np_train)[0], gauss
    )
    loss_train, loss_test, ListeZ1 = utils.train_finding(
        sig_xx_target,
        sig_yy_target,
        sig_xy_target,
        boundary,
        model,
        n_epochs,
        learn_rate,
        scheduler_apply,
        scheduler_gamma=0.5,
    )
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, apply_crack_bounds=True)
