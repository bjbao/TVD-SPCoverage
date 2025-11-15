# Copyright (C) 2025 Brandon Bao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pyvoro
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import (
    mark_inset,
    zoomed_inset_axes,
    inset_axes,
)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from scipy.spatial import ConvexHull

plt.rcParams["text.usetex"] = False
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.size"] = 10


@lru_cache(1)
def colormap(levels):
    """
    Helper function for contour maps
    """
    ncolors = 256
    color_array = plt.get_cmap("afmhot_r")(range(ncolors))
    map_object = LinearSegmentedColormap.from_list(
        name="Reds_alpha", colors=color_array[: -int(ncolors / 4)]
    )
    plt.register_cmap(cmap=map_object)
    map = plt.cm.get_cmap("Reds_alpha", levels + 1)
    map.set_under((1, 1, 1, 0))
    return map


def plot_voronoi_regions(
    position=None, ax=None, zorder=20, vor=None, linewidth=1, pos=None
):
    """
    Used for debugging
    """

    vertices = []
    if vor is None and position is not None:
        vor = pyvoro.compute_2d_voronoi(position, [[-10, 10], [-10, 10]], dispersion=4)
    elif vor is None and pos is not None:
        try:
            vor = pyvoro.compute_2d_voronoi(pos, [[-1.6, 1.6], [-1, 1]], dispersion=10)
        except:
            print("Agent left, skipping trial")
            return
    lines = []
    for partition in vor:
        vertices.append(np.array(partition["vertices"]))
    lines = []
    for cube in vertices:
        hull = ConvexHull(cube)
        for s in hull.simplices:
            a = [np.append(lists, 0) for lists in cube[s]]
            lines.append(a)
    return lines


def plot_cost(folder, filename):
    path = os.path.join(folder, filename)
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    cost = []
    time = []
    for row in data:
        time.append(row[0])
        cost.append(row[1])
    dpi = 600
    fs = (3.5, 3.5)
    cost_val = sum(cost)
    fig = plt.figure(figsize=fs, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(cost)
    fig.suptitle(
        f"{os.path.basename(path)[:-4]} Total Cost {cost_val:.2f}", fontsize=10
    )
    plt.ylabel("Cost H", fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.xlim((0, len(time)))
    plt.ylim(bottom=0)
    x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
    # axins = inset_axes(ax, width=2, height=1.5, loc="lower center")
    # # axins = zoomed_inset_axes(ax, zoom=2, loc='lower center')
    # axins.plot(cost[:200])
    # # fix the number of ticks on the inset axes
    # axins.yaxis.get_major_locator().set_params(nbins=7)
    # axins.xaxis.get_major_locator().set_params(nbins=7)
    # axins.tick_params(labelleft=False, labelbottom=False)
    # mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
    png_name = os.path.join(
        folder, f'Robotarium {filename[:-4].replace("_cost","")} Cost.png'
    )
    print(f"Saved: {png_name}")
    plt.savefig(png_name, bbox_inches="tight", dpi=600)
    # plt.savefig('Robotarium TVD-SP-hybrid'+ "Cost" +
    # ".png", bbox_inches='tight', dpi=600)
    # plt.savefig("pycoverage_robotarium\\CostTVD-SP-hybrid_A_italic_1.png")
    # plt.savefig("pycoverage_robotarium\\CostTVD-SP-hybrid_A_non-italic_1.png")
    # plt.savefig("pycoverage_robotarium\\CostTVD-SP-hybrid_old_A_italic.png")
    # plt.savefig("pycoverage_robotarium\\CostTVD-SP-hybrid_old_A_non-italic.png")
    # plt.savefig("pycoverage_robotarium\\CostTVD-C.png")
    # plt.savefig("pycoverage_robotarium\\CostTVD-SP_1.png")
    return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, cost, label="Cost")
    ax.legend(loc="best")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost H")
    config = f"Algorithm: {CoverageControl.algorithm} Total Cost: {sum(cost):.2f}, {n_agents} agents, {num_points:.0f} samples, {len(time)} iterations"
    ax.set_title("TVD-SP-Cost ")


def main():
    # filename = 'pycoverage_robotarium\\TVD-SP-hybrid_cost_A_italic_1.txt'
    # filename = 'pycoverage_robotarium\\TVD-SP-hybrid_cost_A_non-italic_1.txt'
    # filename = 'pycoverage_robotarium\\TVD-SP-hybrid_old_cost_A_italic.txt'
    # filename = 'pycoverage_robotarium\\TVD-SP-hybrid_old_cost_A_non-italic.txt'
    # filename = 'pycoverage_robotarium\\TVD-C_cost.txt'
    # filename = 'pycoverage_robotarium\\TVD-SP_cost_1.txt'
    folder = "pycoverage_robotarium"
    filenames = [
        "TVD-C_cost.txt",
        "TVD-C_cost_1.txt",
        "TVD-SP-delayed_cost_1.txt",
        "TVD-SP-hybrid_cost_A_italic.txt",
        "TVD-SP-hybrid_cost_A_italic_1.txt",
        "TVD-SP-hybrid_cost_A_non-italic.txt",
        "TVD-SP-hybrid_cost_A_non-italic_1.txt",
        "TVD-SP-hybrid_cost_A_non-italic_2.txt",
        "TVD-SP-hybrid_old_cost_A_italic.txt",
        "TVD-SP-hybrid_old_cost_A_italic_1.txt",
        "TVD-SP-hybrid_old_cost_A_italic_2.txt",
        "TVD-SP-hybrid_old_cost_A_non-italic.txt",
        "TVD-SP-hybrid_old_cost_A_non-italic_1.txt",
        "TVD-SP_cost_1.txt",
        "TVD-SP_cost_2.txt",
    ]
    filenames = ["Lloyd_cost.txt", "TVD-SP-hybrid_cost.txt", "TVD-C_cost_test.txt"]
    for filename in filenames:
        plot_cost(folder, filename)


if __name__ == "__main__":
    main()
