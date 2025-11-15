# Copyright (C) 2020 Melcior Pijoan Comas
#
# This file is part of tvd-coverage.
#
# tvd-coverage is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
TVD-SP paper sim
"""

import matplotlib.pyplot as plt
import numpy as np
from coveragecontrol_2d import Agent, Lloyd, TVD, TVD_K, TVD_SP
from utils_2d import samplePoints
from tqdm import tqdm


def getAlgorithm(algorithm, params, eps, deta, phi, Agents, k):
    if algorithm == "Lloyd":
        CoverageControl = Lloyd(params, phi, Agents)
    elif algorithm == "TVD-C":
        CoverageControl = TVD(params, phi, Agents)
    elif algorithm == "TVD-K":
        CoverageControl = TVD_K(params, k, phi, Agents)
    elif algorithm == "TVD-SP":
        CoverageControl = TVD_SP(params, eps, deta, algorithm, phi, Agents)
    elif algorithm == "TVD-SP-delayed":
        CoverageControl = TVD_SP(params, eps, deta, algorithm, phi, Agents)
    elif algorithm == "TVD-SP-hybrid_italic":
        CoverageControl = TVD_SP(params, eps, deta, algorithm, phi, Agents)
    elif algorithm == "TVD-SP-hybrid_non-italic":
        CoverageControl = TVD_SP(params, eps, deta, algorithm, phi, Agents)
    else:
        raise ValueError("Pick algorithm in algorithm_list")
    return CoverageControl


def run_sim(
    num_points=1e4,
    n_agents=10,
    algorithm="Lloyd",
    T=1000,
    tau=2000,
    phi=4,
    dir=dir,
    eps=1e-3,
    k=1,
):
    """
    Simulation Settings:

    position: Initial agent position. Supports many agents.
    env: Environment bounds, limited to rectangular prism shape.
    mu_a: Initial start position of target agent.
    mu_b: Final position in simulation. The generator will make linear path between the points.
    sigma_a, sigma_b: Covariance at start and end. Generator linearly interpolates value at each frame.
    num_frames: Number of frames for sim. Will affect red agent speed.
    """

    # Parameters for coverage control
    dim = 2
    x_max = 5
    y_max = 5
    env = [
        [-x_max, x_max],
        [-y_max, y_max],
    ]

    n_agents = n_agents
    seed = 1

    # np.random.seed(seed)
    # Initial Conditions to Avoid Barrier Use in the Beginning. (x,y pos for each)
    agent_pos = samplePoints([-x_max, x_max, -y_max, y_max], n_agents, seed)
    print(f"p0:{agent_pos}")
    # agent_pos_init = np.array(
    #     [
    #         [1, 0.5, -0.5, -0.25, 0.28, 0.85, -1, -0.65, -0.8, 1],
    #         [0.8, -0.3, -0.75, 0.1, 0.34, -0.7, 0.7, -0.2, 0.2, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     ]
    # ).T
    Agents = []
    for n in range(n_agents):
        Agents.append(Agent(agent_pos[n, :]))
    Agents = np.array(Agents)
    trim_vel = True

    # Total time T and time step dt
    T = T
    dt = 0.1  # needs to be 1 for sim. 1 will be 0.033 seconds
    t = 0
    # 0.4 m/s max speed
    num_frames = int(np.ceil((T) / dt))
    # num_frames = 2
    # Proportional term to drive agents to centroid
    kappa = 1

    # Coverage control sim config
    max_vel = 3
    max_vel_alg = 3
    # mass_eps = 0  # 3e-3  # condition of no mass in a Voronoi cell, 3 sigma
    num_points = num_points  # Monte Carlo Sampling Points
    # TVD Parameters
    # Number of hops for distributed TVD-K algorithm
    # k = 2

    # Singular Perturbation time scaling parameter, must be small ie: (1e-2 to 1e-6). For larger eps, try TVD_SSP. deta> 1/L where L is the largest
    # |eigval| of the Hessian of the A matrix. A = np.eye(np.size(blue_pos)) -dcdp.
    # eps = 1e-2
    deta = 1

    params = {
        "env": env,
        "trim_vel": trim_vel,
        "seed": seed,
        "T": T,
        "dt": dt,
        "t": t,
        "num_frames": num_frames,
        "kappa": kappa,
        "max_vel": max_vel_alg,
        # "mass_eps": mass_eps,
        "num_points": num_points,
        "tau": tau,
        "phi": phi,
    }
    save_params = (
        "agents"
        + str(n_agents)
        + "eps"
        + str(eps)
        + "deta"
        + str(deta)
        + "max_u"
        + str(max_vel)
        + "T"
        + str(T)
        + "dt"
        + str(dt)
        + "samples"
        + str(num_points)
        + "tau"
        + str(tau)
        + "phi"
        + str(phi)
    )
    if dim == 2:
        save_params += "dim" + str(dim)

    CoverageControl = getAlgorithm(algorithm, params, eps, deta, phi, Agents, k)

    model = 2
    save_params = (
        "agents"
        + str(n_agents)
        + "eps"
        + str(eps)
        + "deta"
        + str(deta)
        + "max_u"
        + str(max_vel)
        + "T"
        + str(T)
        + "dt"
        + str(dt)
        + "samples"
        + str(num_points)
        + "tau"
        + str(tau)
        + "phi"
        + str(phi)
        + "k"
        + str(k)
    )

    for t in tqdm(range(0, num_frames)):
        velocity = CoverageControl.calculateVelocities()
    CoverageControl.saveData(dir, seed, model, save_params)

    # Call at end of script to print debug information and for your script to run on the Robotarium server properly
    cost = CoverageControl.cost_vec
    time = CoverageControl.t_vec

    config = f"Algorithm: {CoverageControl.algorithm} Total Cost: {dt*sum(cost):.2f}, {n_agents} agents, {num_points:.0f} samples, {len(time)} iterations, phi {phi}, deta=1/L"

    print(config)
    name = f"{dir}Algorithm{CoverageControl.algorithm}_TotalCost{dt*sum(cost):.2f}_agents{n_agents}__iterations{len(time):.2f}_phi{phi}_eps{eps}_k{k}"

    np.savetxt(
        f"{name}_cost.txt",
        np.array(list(zip(time, cost))),
        delimiter=",",
        header="time, cost",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, cost, label="Cost")
    ax.legend(loc="best")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cost")
    ax.set_title(config)
    plt.savefig(
        f"{dir}Algorithm{CoverageControl.algorithm}_TotalCost{dt*sum(cost):.2f}_agents{n_agents}_iterations{len(time):.2f}_phi{phi}_eps{eps}_k{k}.png"
    )


def main():
    num_points = 1e4
    n_agents = 10
    dir = "pycoverage2d/data/"
    algorithm_list = [
        # "Lloyd",
        # "TVD-K",
        # "TVD-C",
        # "TVD-SP",
        "TVD-SP-delayed",
        # "TVD-SP-hybrid_italic",
        # "TVD-SP-hybrid_non-italic",
    ]
    T = 10 * np.pi
    tau = 5
    eps_vec = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    k_vec = [0, 1, 2, 3]
    # for algorithm in algorithm_list:
    #     run_sim(num_points, n_agents, algorithm, T=T, tau=tau, phi=1, dir=dir)
    # exit()
    for phi in range(1, 4):
        for algorithm in algorithm_list:
            if "TVD-SP" in algorithm:
                for eps in eps_vec:
                    run_sim(
                        num_points,
                        n_agents,
                        algorithm,
                        T=T,
                        tau=tau,
                        phi=phi,
                        dir=dir,
                        eps=eps,
                    )
            elif "TVD-K" in algorithm:
                for k in k_vec:
                    run_sim(
                        num_points,
                        n_agents,
                        algorithm,
                        T=T,
                        tau=tau,
                        phi=phi,
                        dir=dir,
                        k=k,
                    )
            else:
                run_sim(num_points, n_agents, algorithm, T=T, tau=tau, phi=phi, dir=dir)


if __name__ == "__main__":
    main()
