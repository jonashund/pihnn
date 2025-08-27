"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import torch
import pihnn.geometries as geom
import pihnn.bc as bc
import pihnn_devo.nn as nn_devo
import pihnn_devo.utils as utils_devo

# Network parameters
n_epochs = 600  # Number of epochs
learn_rate = 1e-2  # Initial learning rate
scheduler_apply = [500]  # At which epoch to execute scheduler
units = [1, 10, 10, 10, 1]  # Units in each network layer
np_train = 200  # Number of training points on domain boundary
np_test = 20  # Number of test points on the domain boundary
beta = 0.5  # Initialization parameter
gauss = 3  # Initialization parameter

h = 10  # Half-height of the domain
l = 10  # Half-length of the domain

# Applied stresses at the top and bottom (pure vertical traction/compression)
sig_ext_t = 1j
sig_ext_b = -1j

sig_xx_target = torch.tensor(
    [
        -1.1013e-01,
        -7.8078e-02,
        2.9673e-02,
        -8.2760e-05,
        -1.5666e-01,
        -1.8831e-01,
        8.3253e-01,
        2.9214e-01,
        2.9209e-01,
        8.3255e-01,
        -1.8829e-01,
        -1.5662e-01,
        -1.5070e-04,
        2.9627e-02,
        -7.8077e-02,
        -1.1011e-01,
    ]
)
sig_yy_target = torch.tensor(
    [
        0.9746,
        0.8055,
        0.8462,
        1.2329,
        1.0684,
        0.8392,
        0.8325,
        1.2270,
        1.2270,
        0.8326,
        0.8392,
        1.0684,
        1.2330,
        0.8463,
        0.8055,
        0.9746,
    ]
)
sig_xy_target = torch.tensor(
    [
        1.0912e-01,
        1.5553e-01,
        -2.9327e-01,
        -7.9474e-04,
        7.7158e-02,
        1.8727e-01,
        8.3253e-01,
        -3.0714e-02,
        -3.0711e-02,
        8.3255e-01,
        1.8730e-01,
        7.7203e-02,
        -7.7285e-04,
        -2.9323e-01,
        1.5558e-01,
        1.0919e-01,
    ]
)

line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
# line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.displacement_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

z1 = -0 - 0j
z2 = 3.5 + 3.5j

z1N = z1 / (l)
z2N = z2 / (l)

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
n_runs = 20  # number of repetitions

final_losses_train = []
final_losses_test = []
final_z1 = []
final_z2 = []

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

    model = nn_devo.enriched_PIHNN_devo("km", units, boundary)
    model.initialize_weights(
        "exp", beta, boundary.extract_points(10 * np_train)[0], gauss
    )

    loss_train, loss_test, _ = utils_devo.train_devo(
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
