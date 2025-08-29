# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:24:58 2025

@author: Nicolas
"""

"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""
import sys

# sys.path.append(r"C:\Windows\System32\site-packages\python3.10")
sys.path.append(r"C:\Users\Nicolas\Desktop\Stage_2A\PIHNN")


import pihnn.nn_PIHNN as nn
import pihnn.utils_PIHNN as utils
import pihnn.geometries_PIHNN as geom
import pihnn.graphics_PIHNN as graphics
import pihnn.bc_PIHNN as bc

import torch
import random

# Network parameters
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


# - 3 - 3j , 3 + 3j

# sig_xx_target = torch.tensor([-0.0003, -0.1139, -0.1139, -0.0003])
# sig_yy_target = torch.tensor([1.2515, 0.9721, 0.9721, 1.2516])
# sig_xy_target = torch.tensor([-0.0002,  0.1134,  0.1134, -0.0003])


# - 3 + 0j , 3 + 0j

sig_xx_target = torch.tensor([-0.0738, -0.0738, -0.0738, -0.0738])
sig_yy_target = torch.tensor([1.0509, 1.0509, 1.0509, 1.0509])
sig_xy_target = torch.tensor([-0.1081, 0.1081, 0.1081, -0.1081])


# - 2  -2j , 2 + 2j


# sig_xx_target = torch.tensor([ 8.6255e-05, -6.3983e-02, -6.4015e-02,  9.8056e-05])
# sig_yy_target = torch.tensor([1.0915, 0.9928, 0.9928, 1.0914])
# sig_xy_target = torch.tensor([-0.0002,  0.0639,  0.0638, -0.0002])

#############################################################################

# - 3 - 0j , 3 + 0j

# sig_xx_target = torch.tensor([-0.0553, -0.0054, -0.0053, -0.0550, -0.0525, -0.1066, -0.1064, -0.0521,
#         -0.0524, -0.1066, -0.1065, -0.0521, -0.0552, -0.0054, -0.0053, -0.0550])
# sig_yy_target = torch.tensor([1.0444, 0.8817, 0.8817, 1.0444, 1.1595, 0.9481, 0.9482, 1.1595, 1.1595,
#         0.9482, 0.9481, 1.1595, 1.0444, 0.8817, 0.8817, 1.0444])
# sig_xy_target = torch.tensor([-0.0720, -0.1372,  0.1378,  0.0725, -0.0093, -0.2979,  0.2983,  0.0097,
#          0.0095,  0.2982, -0.2981, -0.0094,  0.0722,  0.1374, -0.1375, -0.0722])


# z1 = complex(random.uniform(-9, 9), random.uniform(-9, 9))
# z2 = complex(random.uniform(-9, 9), random.uniform(-9, 9))

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
# crack = geom.line(P1= z1, P2= z2, bc_type=bc.stress_bc())

crack.add_crack_tip(tip_side=0)  # Left tip
crack.add_crack_tip(tip_side=1)  # Right tip


boundary = geom.boundary(
    [line1, line2, line3, line4, crack], np_train, np_test, enrichment="rice"
)


# === Training loop over multiple runs ===
if __name__ == "__main__":
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
        # crack = geom.line(P1=z1, P2=z2, bc_type=bc.stress_bc())
        crack.add_crack_tip(tip_side=0)  # Left tip
        crack.add_crack_tip(tip_side=1)  # Right tip

        boundary = geom.boundary(
            [line1, line2, line3, line4, crack], np_train, np_test, enrichment="rice"
        )

        model = nn.enriched_PIHNN_finding("km", units, boundary)
        model.initialize_weights(
            "exp", beta, boundary.extract_points(10 * np_train)[0], gauss
        )

        ########################### optimizer : adam ###################################

        loss_train, loss_test, ListeZ1, ListeZ2 = utils.train_finding_adam(
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

        ########################### optimizer : L_BFGS ###################################

        # loss_train, loss_test, ListeZ1 , ListeZ2 = utils.train_finding_L_BFGS(
        #     sig_xx_target, sig_yy_target, sig_xy_target,
        #     boundary, model, n_epochs, learn_rate, dir="results/", apply_adaptive_sampling=None)

        #################################################################################

        # Keep the last values
        final_losses_train.append(loss_train[-1])
        final_losses_test.append(loss_test[-1])

        z1_val = model.z1.detach().item() * l
        z2_val = model.z2.detach().item() * l

        # z1_val = model.z1.detach().item()
        # z2_val = model.z2.detach().item()

        final_z1.append(z1_val)
        final_z2.append(z2_val)

        print("z1:", z1_val)
        print("z2:", z2_val)
        print("Final training loss:", loss_train[-1])
        print("Final test loss:", loss_test[-1])

        # Check if this is the best test loss
        if loss_test[-1] < best_test_loss:
            best_test_loss = loss_test[-1]
            best_z1 = z1_val
            best_z2 = z2_val
            best_run = i + 1

    # Averages
    avg_train_loss = sum(final_losses_train) / len(final_losses_train)
    avg_test_loss = sum(final_losses_test) / len(final_losses_test)
    avg_z1 = sum(final_z1) / len(final_z1)
    avg_z2 = sum(final_z2) / len(final_z2)

    print("\n=== Results after {} runs ===".format(n_runs))
    print("Average final training loss:", avg_train_loss)
    print("Average final test loss:", avg_test_loss)
    print("Average z1:", avg_z1)
    print("Average z2:", avg_z2)

    print("\n=== Best run (based on test loss) ===")
    print("Best run index:", best_run, "/", n_runs)
    print("Best test loss:", best_test_loss)
    print("Best z1:", best_z1)
    print("Best z2:", best_z2)
