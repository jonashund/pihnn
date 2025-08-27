import torch
import pihnn.nn as nn
import pihnn.bc as bc
import pihnn.utils as utils
import numpy as np
from tqdm import tqdm
import os


def PIHNNloss_devo(sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, t):
    """
    Combined loss = boundary_loss (km_loss) + domain/points_loss (data_loss)
    """
    if model.PDE in ["km", "km-so"]:
        loss_bc = utils.km_loss(boundary, model, t)
        loss_data = data_loss(
            sig_xx_target=sig_xx_target,
            sig_yy_target=sig_yy_target,
            sig_xy_target=sig_xy_target,
            model=model,
        )
        loss_reg = reg_loss(model=model)
        print("loss_bc : ", loss_bc)
        print("loss_data : ", 1.0e-2 * loss_data)
        print("loss_reg : ", loss_reg)

        return 1.0e0 * loss_bc + 1.0e-2 * loss_data + 1.0e2 * loss_reg
    elif model.PDE in ["laplace", "biharmonic"]:
        return utils.scalar_loss(boundary, model, t)
    else:
        raise ValueError(
            "'model.PDE' must be either 'laplace', 'biharmonic', 'km' or 'km-so'."
        )


def data_loss(sig_xx_target, sig_yy_target, sig_xy_target, model, weight_xy=1):
    """
    Compute the loss (MSE) on fixed data points, weighted by sigma_xy.

    :sig_xx_target: target values for sigma_xx at data points
    :sig_yy_target: target values for sigma_yy at data points
    :sig_xy_target: target values for sigma_xy at data points
    :model: neural network model
    :param weight_xy: weight for sigma_xy, optional, default is 1
    :returns: MSE loss value
    """
    l = 10

    x_vals = torch.tensor([-6.0, -3.0, 3.0, 6.0])
    y_vals = torch.tensor([6.0, 3.0, -3.0, -6.0])  # ajout de -3.0 et 3.0

    # Normalisation
    x_vals_norm = x_vals / l
    y_vals_norm = y_vals / l

    # Construction de z_data avec les valeurs normalisées
    z_data = torch.cat([x_vals_norm + 1j * y for y in y_vals_norm])
    z_data = z_data.to(torch.cfloat).requires_grad_(True)

    sig_xx, sig_yy, sig_xy, _, _ = model(z_data, real_output=True)

    # Weighted MSE loss
    L = utils.MSE(sig_xx, sig_xx_target)
    L += utils.MSE(sig_yy, sig_yy_target)
    L += weight_xy * utils.MSE(sig_xy, sig_xy_target)

    return L / z_data.nelement()


def reg_loss(model, limit=1, weight=100.0):
    """
    Penalize z1 and z2 if they fall outside the square [-limit, limit] x [-limit, limit].
    The penalty is zero if z1 and z2 are inside the square.
    Otherwise, the penalty is quadratic.
    """
    penalties = []

    for z in [model.z1, model.z2]:
        x = z.real
        y = z.imag

        if -limit <= x <= limit and -limit <= y <= limit:
            penalties.append(torch.tensor(0.0, dtype=x.dtype, device=x.device))
        else:
            penalty_x = torch.relu(x - limit) + torch.relu(-limit - x)
            penalty_y = torch.relu(y - limit) + torch.relu(-limit - y)
            penalty = weight * (penalty_x**2 + penalty_y**2)
            penalties.append(penalty)

    return penalties[0] + penalties[1]


def train_devo(
    sig_xx_target,
    sig_yy_target,
    sig_xy_target,
    boundary,
    model,
    n_epochs,
    learn_rate=1e-3,
    scheduler_apply=[],
    scheduler_gamma=0.5,
    dir="results/",
    apply_adaptive_sampling=0,
):
    """
    Performs the training of the neural network with one moving crack tip (z1).

    :sig_xx_target: target values for sigma_xx at data points.
    :sig_yy_target: target values for sigma_yy at data points.
    :sig_xy_target: target values for sigma_xy at data points.
    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.enriched_PIHNN` or similar
    :param n_epochs: Number of total epochs.
    :type n_epochs: int
    :param learn_rate: Initial learning rate for the optimizer.
    :type learn_rate: float
    :param scheduler_apply: Epochs at which to apply the learning rate scheduler.
    :type scheduler_apply: list of int
    :param scheduler_gamma: Scheduler decay factor.
    :type scheduler_gamma: float
    :param dir: Directory where outputs will be saved.
    :type dir: str
    :param apply_adaptive_sampling: Epoch at which to apply adaptive sampling (0 = never).
    :type apply_adaptive_sampling: int
    :returns: Tuple of two lists (loss_epochs, loss_epochs_test)
    """

    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if name not in ["z1", "z2"]
                ],
                "lr": learn_rate,
            },
            {"params": [model.z1], "lr": 10 * learn_rate},
            {"params": [model.z2], "lr": 10 * learn_rate},
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    loss_epochs = []
    loss_epochs_test = []

    for bc_type in boundary.bc_types:
        if model.PDE in ["laplace", "biharmonic"] and not issubclass(
            bc_type.__class__, bc.scalar_bc
        ):
            raise ValueError(
                "'laplace' and 'biharmonic' problems require boundary conditions derived from 'scalar_bc'."
            )
        elif model.PDE in ["km", "km-so"] and not issubclass(
            bc_type.__class__, bc.linear_elasticity_bc
        ):
            raise ValueError(
                "Linear elasticity problems require boundary conditions derived from 'linear_elasticity_bc'."
            )

    z1_list = []
    z2_list = []
    for epoch_id in (pbar := tqdm(range(n_epochs))):

        z1 = complex(model.z1)
        z2 = complex(model.z2)

        model.update_boundary()

        z1_list.append(z1)
        z2_list.append(z2)

        print("z1 : ", z1)
        print("z2 : ", z2)

        if epoch_id == apply_adaptive_sampling and epoch_id != 0:
            utils.RAD_sampling(boundary, model)

        optimizer.zero_grad()
        model.zero_grad()

        loss = PIHNNloss_devo(
            sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, "training"
        )
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_epochs.append(loss.cpu().item())
        loss_test = PIHNNloss_devo(
            sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, "test"
        )
        loss_epochs_test.append(loss_test.cpu().item())

        with torch.autograd.no_grad():
            if epoch_id % 10 == 0:
                pbar.set_postfix_str(
                    f"train: {loss_epochs[-1]:.2E}, test: {loss_epochs_test[-1]:.2E}"
                )
            if epoch_id in scheduler_apply:
                scheduler.step()

    loss = np.column_stack((loss_epochs, loss_epochs_test))
    if not dir.endswith("/"):
        dir += "/"
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("# Created path " + os.path.abspath(dir))

    np.savetxt(dir + "loss.dat", loss)
    print("# Saved loss at " + os.path.abspath(dir + "loss.dat"))
    torch.save(model.state_dict(), dir + "model.dict")
    print("# Saved neural network model at " + os.path.abspath(dir + "model.dict"))

    return loss_epochs, loss_epochs_test, z1_list
