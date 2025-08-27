import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.use("Agg")


def plot_crack(model, epoch=None, ax=None, color="red", show=True, save_path=None):
    """
    Display the crack defined by the two tips z1 and z2 in the model.

    :param model: enriched_PIHNN_finding model with z1 and z2 as parameters.
    :param epoch: epoch number (optional, for title and file name, default is None).
    :param ax: existing matplotlib axes (optional, default is None).
    :param color: crack colour (optional, default is "red").
    :param show: Boolean to show the figure (default is True).
    :param save_path: path to save the image (without file extension), (optional default is None to display plot).
    """
    z1 = model.z1.detach().cpu().numpy()
    z2 = model.z2.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot([z1.real, z2.real], [z1.imag, z2.imag], "-", color=color, linewidth=2)
    ax.scatter([z1.real, z2.real], [z1.imag, z2.imag], color=color, s=30)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    title = f"Crack tip evolution"
    if epoch is not None:
        title += f" (epoch {epoch})"
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        fname = (
            f"{save_path}_epoch_{epoch:04d}.png"
            if epoch is not None
            else f"{save_path}.png"
        )
        plt.savefig(fname)
    if show and ax is None:
        plt.show()
    elif not show:
        plt.close()
