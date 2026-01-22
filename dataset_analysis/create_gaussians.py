import numpy as np
import matplotlib.pyplot as plt

def plot_gaussians(
    gaussians,
    x_range=10,
    resolution=1200,
    filename="gaussians",
    linewidth=2
):
    """
    Plot any number of Gaussian curves and export as transparent SVG.

    Parameters:
        gaussians : list of dicts, each with keys:
            - mu
            - sigma
            - height
        x_range   : Domain size
        resolution: Number of samples
        filename  : Output SVG filename
        linewidth : Line thickness
    """

    x = np.linspace(-x_range, x_range, resolution)

    fig = plt.figure(figsize=(8, 2.5), facecolor="none")
    ax = plt.gca()

    for g in gaussians:
        mu = g["mu"]
        sigma = g["sigma"]
        height = g.get("height", 1.0)

        y = height * np.exp(-(x - mu)**2 / (2 * sigma**2))
        plt.plot(x, y, linewidth=linewidth, color='black')

    # Clean horizontal axis only
    # plt.axhline(0, linewidth=1)
    plt.axis("off")
    plt.margins(x=0)

    # Clean horizontal axis only
    #plt.axhline(0, linewidth=1)
    plt.axis("off")
    plt.margins(x=0)

    plt.savefig(filename + '.png', bbox_inches='tight', dpi=300)
    plt.savefig(filename + '.svg', transparent=True, bbox_inches="tight")
    plt.savefig(filename + '.pdf', bbox_inches="tight")

    plt.show()


# # endoscopic IQA
gaussians = [
    {"mu": -3, "sigma": 1.5, "height": 0.9},
    {"mu": 3, "sigma": 0.7, "height": 1.5},
]
plot_gaussians(
    gaussians,
    x_range=10,
    filename="endoscopic_IQA_gaussians",
    linewidth=2
)

#
# # # endoscopic IQA close
# gaussians = [
#     {"mu": -0.5, "sigma": 1.5, "height": 1.5},
#     {"mu": 0.5, "sigma": 1.5, "height": 1.5},
# ]
# plot_gaussians(
#     gaussians,
#     x_range=10,
#     filename="endoscopic_IQA_close_gaussians",
#     linewidth=2
# )

# # prevelance shift
# gaussians = [
#     {"mu": 0, "sigma": 2.0, "height": 1.2},
#     {"mu": -5, "sigma": 1.0, "height": 0.3},
# ]
# # prevelance shift
# plot_gaussians(
#     gaussians,
#     filename="prevelance_shift_gaussians",
#     x_range=10,
#     linewidth=2
# )

# # surgical procedures
# gaussians = [
#     {"mu": 0, "sigma": 1.0, "height": 2.0},
#     {"mu": -3.5, "sigma": 0.5, "height": 0.5},
#     {"mu": 3.5, "sigma": 0.5, "height": 0.5},
#     {"mu": -7, "sigma": 0.3, "height": 0.3}
# ]
#
# plot_gaussians(
#     gaussians,
#     x_range=10,
#     filename="surgical_procedures_gaussians",
#     linewidth=2
# )


# # surgical scene understanding
# gaussians = [
#     {"mu": 0, "sigma": 3.5, "height": 2.0},
#     {"mu": 0, "sigma": 1.0, "height": 2.0},
#     {"mu": -3.5, "sigma": 0.5, "height": 0.5},
#     {"mu": 3.5, "sigma": 0.5, "height": 0.5},
#     {"mu": -7, "sigma": 0.3, "height": 0.3}
# ]
#
# plot_gaussians(
#     gaussians,
#     x_range=10,
#     filename="surgical_scene_understanding_gaussians",
#     linewidth=2
# )