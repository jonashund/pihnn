"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import os
import numpy as np
import torch
import pihnn.geometries as geom
import pihnn.bc as bc
import pihnn_devo.nn as nn_devo
import pihnn_devo.utils as utils_devo

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # set environment variable to avoid error message
)

# Network parameters
# n_epochs = 5  # Number of epochs
n_epochs = 500  # Number of epochs
learn_rate = 1e-3  # Initial learning rate
scheduler_apply = []  # At which epoch to execute scheduler
units = [1, 10, 10, 10, 1]  # Units in each network layer
np_train = 150  # Number of training points on domain boundary
np_test = 10  # Number of test points on the domain boundary
beta = 0.5  # Initialization parameter
gauss = 3  # Initialization parameter

h = 10  # Half-height of the domain
l = 10  # Half-length of the domain

# Applied stresses at the top and bottom (pure vertical traction/compression)
sig_ext_t = 1j
sig_ext_b = -1j

# import stress data and coordinates from text files
data_dir = "../test/collect_data_pihnn/"
stress_file = data_dir + "stress_data.txt"
coords_file = data_dir + "coords_data.txt"
stress_data = np.loadtxt(stress_file, skiprows=1)  # Skip header row
coords_data = np.loadtxt(coords_file, skiprows=1)

stress_data = torch.Tensor(stress_data)
coords_data = torch.Tensor(coords_data)

z1 = 0 - 0j
z2 = 3 + 0j

z1N = z1 / (l)
z2N = z2 / (l)

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
# line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.displacement_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

crack = geom.line(P1=z1N, P2=z2N, bc_type=bc.stress_bc())

crack.add_crack_tip(tip_side=0)  # Left tip
crack.add_crack_tip(tip_side=1)  # Right tip

boundary = geom.boundary(
    curves=[line1, line2, line3, line4, crack],
    np_train=np_train,
    np_test=np_test,
    enrichment="rice",
)

# === Training loop over multiple runs ===
n_runs = 10  # number of repetitions

final_losses_train = []
final_losses_test = []
final_z1 = []
final_z2 = []

best_test_loss = float("inf")
best_z1 = None
best_z2 = None
best_run = -1

for i in range(n_runs):
    print("\n=== Run {} / {} ===".format(i + 1, n_runs))

    # We must re-instantiate a fresh model at each run
    crack = geom.line(P1=z1N, P2=z2N, bc_type=bc.stress_bc())

    crack.add_crack_tip(tip_side=0)  # Left tip
    crack.add_crack_tip(tip_side=1)  # Right tip

    boundary = geom.boundary(
        curves=[line1, line2, line3, line4, crack],
        np_train=np_train,
        np_test=np_test,
        enrichment="rice",
    )

    model = nn_devo.enriched_pihnn_devo("km", units, boundary)
    model.initialize_weights(
        "exp", beta, boundary.extract_points(10 * np_train)[0], gauss
    )

    loss_train, loss_test, z1_list, z2_list = utils_devo.train_devo_adam(
        sig_xx_target=stress_data[:, 0],
        sig_yy_target=stress_data[:, 1],
        sig_xy_target=stress_data[:, 2],
        x_coords=coords_data[0],
        y_coords=coords_data[1],
        boundary=boundary,
        model=model,
        n_epochs=n_epochs,
        learn_rate=learn_rate,
        scheduler_apply=scheduler_apply,
        scheduler_gamma=0.5,
        dir="../test/find_cracks_pihnn/",
    )

    # We keep the last values
    final_losses_train.append(loss_train[-1])
    final_losses_test.append(loss_test[-1])

    z1_val = model.z1.detach().item() * l
    z2_val = model.z2.detach().item() * l
    final_z1.append(z1_val)
    final_z2.append(z2_val)

    print("z1 : ", z1_val)
    print("z2 : ", z2_val)
    print("Final train loss:", loss_train[-1])
    print("Final test loss :", loss_test[-1])

    # Check if this is the best test loss
    if loss_test[-1] < best_test_loss:
        best_test_loss = loss_test[-1]
        best_z1 = z1_val
        best_z2 = z2_val
        best_run = i + 1

# Average of the final losses
avg_train_loss = sum(final_losses_train) / len(final_losses_train)
avg_test_loss = sum(final_losses_test) / len(final_losses_test)

# Average of z1 and z2
avg_z1 = sum(final_z1) / len(final_z1)
avg_z2 = sum(final_z2) / len(final_z2)

print("\n=== Results after {} runs ===".format(n_runs))
print("Average final train loss:", avg_train_loss)
print("Average final test loss :", avg_test_loss)
print("Average z1 :", avg_z1)
print("Average z2 :", avg_z2)
