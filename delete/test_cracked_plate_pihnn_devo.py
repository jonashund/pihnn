# -*- coding: utf-8 -*-
"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import delete.nn_devo as nn
import delete.utils_devo as utils


import torch

# Network parameters
# n_epochs = 6000 # Number of epochs
n_epochs = 200  # TODO: remove after debugging
learn_rate = 1e-5  # Initial learning rate
scheduler_apply = [500, 1000, 1500]
units = [1, 10, 10, 10, 1]  # Units in each network layer
np_train = 300  # Number of training points on domain boundary
np_test = 20  # Number of test points on the domain boundary
beta = 0.5  # Initialization parameter
gauss = 3  # Initalization parameter


# -----------------------------------
# Domaine géométrique et conditions aux limites
# -----------------------------------

h = 10  # Demi-hauteur du domaine
l = 10  # Demi-longueur du domaine
n_segments = 50  # Nombre de segments pour chaque ligne

# Contraintes imposées en haut et en bas (traction/compression pure verticale)
sig_ext_t = 1j  # Traction en haut : σ_yy = +1
sig_ext_b = -1j


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
