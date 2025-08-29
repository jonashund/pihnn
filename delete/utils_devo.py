import torch
import os, inspect, warnings
import numpy as np
import pihnn.bc as bc
import pihnn.nn as nn

from tqdm import tqdm


def PIHNNloss_finding(sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, t):
    """
    Loss combinée = loss aux bords (km_loss) + loss aux points (data_loss)
    """
    if model.PDE in ["km", "km-so"]:
        lossBC = km_loss(boundary, model, t)
        lossData = data_loss(
            sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, t
        )
        lossReg = reg_loss(boundary, model, t)
        print("lossBC : ", lossBC)
        print("lossData : ", 0.01 * lossData)
        print("lossReg : ", lossReg)
        # return lossData + 100*lossReg
        return 1 * lossBC + 0.01 * lossData + 100 * lossReg
    elif model.PDE in ["laplace", "biharmonic"]:
        return scalar_loss(boundary, model, t)  # tu peux ajouter data_loss si besoin
    else:
        raise ValueError(
            "'model.PDE' must be either 'laplace', 'biharmonic', 'km' or 'km-so'."
        )


def get_complex_function(func):
    """
    | Convert the input function to the unified format (i.e., callable: complex :class:`torch.tensor` -> complex :class:`torch.tensor`).
    | The library allows to define some functions in multiple and flexible ways (callables, constants, lists, tensors).
      Then, this method includes all the possible definitions and unify the type of the generic function.

    :param func: Generic input function.
    :type func: int/float/complex/list/tuple/:class:`torch.tensor`/callable
    :returns:
        - **new_func** (callable) - Copy of the input function in the unified format.
    """
    if callable(func):
        if len(inspect.getfullargspec(func)[0]) == 1:
            output = func(torch.tensor([0.0j]))
            if isinstance(output, (int, float, complex)):
                new_func = lambda z: func(z) + 0 * z
            elif torch.is_tensor(output):
                if output.nelement() == 1:
                    new_func = lambda z: func(z) + 0 * z
                elif output.nelement() == 2:
                    new_func = lambda z: func(z)[0] + 1.0j * func(z)[1] + 0 * z
            elif isinstance(output, (tuple, list)):
                new_func = lambda z: func(z)[0] + 1.0j * func(z)[1] + 0 * z
            else:
                raise ValueError(
                    "No suitable combination found for the output of input function."
                )
        elif len(inspect.getfullargspec(func)[0]) == 2:
            output = func(torch.tensor([0.0]), torch.tensor([0.0]))
            if isinstance(output, (int, float, complex)):
                new_func = lambda z: func(z) + 0 * z
            elif torch.is_tensor(output):
                if output.nelement() == 1:
                    new_func = lambda z: func(z.real, z.imag) + 0 * z
                elif output.nelement() == 2:
                    new_func = (
                        lambda z: func(z.real, z.imag)[0]
                        + 1.0j * func(z.real, z.imag)[1]
                        + 0 * z
                    )
            elif isinstance(output, (tuple, list)):
                new_func = (
                    lambda z: func(z.real, z.imag)[0]
                    + 1.0j * func(z.real, z.imag)[1]
                    + 0 * z
                )
            else:
                raise ValueError(
                    "No suitable combination found for the output of input function."
                )
        else:
            raise ValueError(
                "Input function must accept either 1 complex input or 2 real inputs."
            )
    elif isinstance(func, (int, float, complex)):
        new_func = lambda z: func + 0 * z
    elif isinstance(func, (list, tuple, torch.tensor)):
        func_pt = torch.tensor(func)
        if func_pt.nelement() == 1:
            func_pt = func_pt + 0j
        elif func_pt.nelement() == 2:
            func_pt = func_pt[0] + 1j * func_pt[1]
        else:
            raise ValueError(
                "List/tuple/tensor input cannot have more than 2 dimensions."
            )
        new_func = lambda z: func_pt + 0 * z
    else:
        raise ValueError(
            "Input function must be a callable, a scalar, or a pair of values."
        )
    return new_func


def km_loss(boundary, model, t):
    """
    Called by :func:`pihnn.utils.PIHNNloss` if one aims to solve the linear elasticity problem
    through the Kolosov-Muskhelishvili representation.

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param t: Option for 'training' or 'test'.
    :type t: str
    :returns: **loss** (float) - Computed loss.
    """

    if isinstance(model, nn.DD_PIHNN):
        z, normals, bc_idxs, bc_values, mask, twins = boundary(t, dd=True)
    else:
        z, normals, bc_idxs, bc_values = boundary(t)

    vars = model(z.requires_grad_(True), real_output=True)

    if isinstance(model, nn.DD_PIHNN):
        vars[:, twins[0], twins[2]] -= vars[:, twins[1], twins[3]]
        vars[:, twins[1], twins[3]] = 0
        vars[:, mask] = 0

    sig_xx, sig_yy, sig_xy, u_x, u_y = vars

    L = 0.0
    for j, bc_type in enumerate(boundary.bc_types):
        L += MSE(
            bc_type(z, sig_xx, sig_yy, sig_xy, u_x, u_y, normals, bc_values)[
                bc_idxs == j
            ]
        )
    bc.reset_variables()

    return L / z.nelement()


def data_loss(
    sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, t, weight_xy=1
):
    """
    Calcule la perte (MSE) sur des points de données fixés, en pondérant σ_xy.

    :param weight_xy: Pondération pour la composante σ_xy
    """

    # x_vals = torch.tensor([-3.0, -1.5, 1.5, 3.0])
    # y_vals = torch.tensor([3.0, -3.0])
    # z_data = torch.cat([x_vals + 1j * y for y in y_vals])
    # z_data = z_data.to(torch.cfloat).requires_grad_(True)

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

    print(sig_xx.size(), sig_xx_target.size())
    print(sig_xx, sig_xx_target)

    # Calcul pondéré des erreurs
    L = MSE(sig_xx, sig_xx_target)
    L += MSE(sig_yy, sig_yy_target)
    L += weight_xy * MSE(sig_xy, sig_xy_target)

    return L / z_data.nelement()


def reg_loss(boundary, model, t, limit=1, weight=100.0):
    """
    Pénalise z1 et z2 s'ils sortent du carré [-limit, limit] x [-limit, limit].
    La pénalité est nulle si z1 et z2 sont dans le carré.
    Quadratique douce sinon.
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


def scalar_loss(boundary, model, t):
    """
    Called by :func:`pihnn.utils.PIHNNloss` if one aims to solve the Laplace or biharmonic problem.

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param t: Option for 'training' or 'test'.
    :type t: str
    :returns: **loss** (float) - Computed loss.
    """
    if isinstance(model, nn.DD_PIHNN):
        z, normals, bc_idxs, bc_values, mask, twins = boundary(t, dd=True)
    else:
        z, normals, bc_idxs, bc_values = boundary(t)

    if model.PDE in ["laplace", "biharmonic"]:
        u = model(z.requires_grad_(True), real_output=True)
    else:
        raise ValueError(
            "'model.PDE' must be 'laplace' or 'biharmonic' for this type of loss."
        )

    if isinstance(model, nn.DD_PIHNN):
        u[twins[0], twins[2]] -= u[twins[1], twins[3]]
        u[twins[1], twins[3]] = 0
        u[mask] = 0

    L = 0.0
    for j, bc_type in enumerate(boundary.bc_types):
        L += bc_type.scaling_coeff * MSE(
            bc_type(z, u, normals, bc_values)[bc_idxs == j]
        )
    bc.reset_variables()
    return L / z.nelement()


def MSE(value, true_value=None):
    """
    Mean squared error (MSE). Equivalent to torch.nn.MSELoss() except it takes into account empty inputs.

    :param value: Input to evaluate.
    :type value: :class:`torch.tensor`
    :param true_value: Reference value to compare the input with. If unassigned, it's by default zero.
    :type value: :class:`torch.tensor`
    :returns: **mse** (float) - MSE error.
    """
    if value.nelement() == 0:
        return 0.0
    if true_value is None:
        true_value = torch.zeros_like(value)
    return torch.nn.MSELoss(reduction="sum")(value, true_value)


def train_finding(
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
    Performs the training of the neural network with a moving crack tip (z1).

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

    ListeZ1 = []
    ListeZ2 = []
    for epoch_id in (pbar := tqdm(range(n_epochs))):

        z1 = complex(model.z1)
        z2 = complex(model.z2)

        model.update_boundary()

        ListeZ1.append(z1)
        ListeZ2.append(z2)

        print("z1 : ", z1)
        print("z2 : ", z2)

        if epoch_id == apply_adaptive_sampling and epoch_id != 0:
            RAD_sampling(boundary, model)

        optimizer.zero_grad()
        model.zero_grad()

        loss = PIHNNloss_finding(
            sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, "training"
        )
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_epochs.append(loss.cpu().item())
        loss_test = PIHNNloss_finding(
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

    return loss_epochs, loss_epochs_test, ListeZ1
