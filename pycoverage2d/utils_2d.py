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

import cProfile
import functools
import os
import pickle
import pstats
import numpy as np


def profile(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            retval = func(*args, **kwargs)
        finally:
            filename = "profile.prof"  # You can change this if needed
            profiler.dump_stats(filename)
            profiler.disable()
            with open("profile.txt", "w") as profile_file:
                sortby = "cumulative"
                stats = pstats.Stats(profiler, stream=profile_file).sort_stats(sortby)
                stats.print_stats()
        return retval

    return inner


def getPhi(number=4, num_frames=100, dim=2, T=100, tau=5):

    if number == 1:
        mu_a = [10.0, 10.0]
        mu_b = [0.0, 0.0]

        sigma_a = [0.75, 0.75]
        sigma_b = [0.75, 0.75]
        radius = 0.6
        center = np.array([0, 0])
        rotations = T / tau  # L: tau
        a = np.array([1, 0])
        b = np.array([0, 1])
        c = 0.6
        d = 0.6
        trajectory = Ellipse(
            mu_a,
            mu_b,
            sigma_a,
            sigma_b,
            num_frames,
            dim,
            radius,
            center,
            a,
            b,
            c,
            d,
            rotations,
        )
    elif number == 2:
        mu_a = [10.0, 10.0]
        mu_b = [0.0, 0.0]

        sigma_a = [0.75, 0.75]
        sigma_b = [0.75, 0.75]
        radius = 0.6
        center = np.array([0, 0])
        rotations = T / tau  # L: tau
        a = np.array([1, 0])
        b = np.array([0, 1])
        c = 0.6
        d = 0.6
        trajectory = Ellipse(
            mu_a,
            mu_b,
            sigma_a,
            sigma_b,
            num_frames,
            dim,
            radius,
            center,
            a,
            b,
            c,
            d,
            rotations,
        )
    elif number == 3:
        mu_a = [10.0, 10.0]
        mu_b = [0.0, 0.0]

        sigma_a = [0.75, 0.75]
        sigma_b = [0.75, 0.75]
        radius = 0.6
        center = np.array([0, 0])
        rotations = T / tau  # L: tau
        a = np.array([1, 0])
        b = np.array([0, 1])
        c = 0.6
        d = 0.6
        trajectory = Ellipse(
            mu_a,
            mu_b,
            sigma_a,
            sigma_b,
            num_frames,
            dim,
            radius,
            center,
            a,
            b,
            c,
            d,
            rotations,
        )
    elif number == 4:
        mu_a = [10.0, 10.0]
        mu_b = [0.0, 0.0]
        sigma_a = np.array([[0.3, 0.3], [0.6, 0.6]])
        sigma_b = np.array([[0.3, 0.3], [0.6, 0.6]])
        radius = np.array([0.3, 0.3])
        center = np.array([[-1.6 / 3.0, 0], [1.6 / 3, 0]])
        rotations = T / tau  # L: tau
        a = np.array([[1, 0], [1, 0]])
        b = np.array([[0, 1], [0, 1]])
        c = np.array([0.3, 0.3])
        d = np.array([0.3, 0.3])
        trajectory = DoubleEllipse(
            mu_a,
            mu_b,
            sigma_a,
            sigma_b,
            num_frames,
            dim,
            radius,
            center,
            a,
            b,
            c,
            d,
            rotations,
        )
    else:
        raise ValueError("Pick phi 1-4")
    return trajectory


def samplePoints(bounding_box, n_agents, seed, scale=1):
    np.random.seed(seed)
    p0 = np.zeros([n_agents, 2])
    bounding_box = scale * np.array(bounding_box)
    p0[:, 0] = bounding_box[0] + (bounding_box[1] - bounding_box[0]) * np.random.rand(
        n_agents
    )
    p0[:, 1] = bounding_box[2] + (bounding_box[3] - bounding_box[2]) * np.random.rand(
        n_agents
    )
    return p0


def save_data(dir, seed, model, algorithm, params, data):
    if not os.path.exists(dir):
        os.makedirs(dir)

    name = dir + algorithm + params + "phi_" + str(model) + "_seed" + str(seed) + ".pkl"
    with open(name, "wb") as f:
        pickle.dump(data, f)


def load_data(dir, seed, model, algorithm, params):
    name = dir + algorithm + params + "phi_" + str(model) + "_seed" + str(seed) + ".pkl"
    with open(name, "rb") as f:
        data = pickle.load(f)

    return data


class Trajectory:
    """
    Create target agent trajectories. One agent has a position mu and covariance sigma, start and end points, and the trajectory runs for a num_frames
    """

    def __init__(self, mu_a, mu_b, sigma_a, sigma_b, num_frames, dim):
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.num_frames = num_frames
        self.x = None
        self.y = None
        self.sigma_xt = None
        self.sigma_yt = None
        self.dim = dim
        self.sigmaGenerator()

    def sigmaGenerator(self):
        self.sigma_xt = []
        self.sigma_yt = []
        for t in range(self.num_frames):
            self.sigma_xt.append(
                self.sigma_a[0] * (1 - float(t) / self.num_frames)
                + self.sigma_b[0] * (float(t) / self.num_frames)
            )

            self.sigma_yt.append(
                self.sigma_a[1] * (1 - float(t) / self.num_frames)
                + self.sigma_b[1] * (float(t) / self.num_frames)
            )

    def getValuesAtT(self, i):
        if i < 1:
            mu = np.array(
                [
                    [
                        self.x[0],
                        self.y[0],
                    ],
                    [
                        self.x[0],
                        self.y[0],
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        self.sigma_xt[0],
                        self.sigma_yt[0],
                    ],
                    [
                        self.sigma_xt[0],
                        self.sigma_yt[0],
                    ],
                ]
            )
        else:
            mu = np.array(
                [
                    [
                        self.x[i],
                        self.y[i],
                    ],
                    [
                        self.x[i - 1],
                        self.y[i - 1],
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        self.sigma_xt[i],
                        self.sigma_yt[i],
                    ],
                    [
                        self.sigma_xt[i - 1],
                        self.sigma_yt[i - 1],
                    ],
                ]
            )
        return mu, sigma


class Ellipse(Trajectory):
    def __init__(
        self,
        mu_a,
        mu_b,
        sigma_a,
        sigma_b,
        num_frames,
        dim,
        radius,
        center,
        a,
        b,
        c,
        d,
        rotations,
    ):
        super().__init__(mu_a, mu_b, sigma_a, sigma_b, num_frames, dim)
        self.radius = radius
        self.center = center
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.rotations = rotations
        self.calculateEllipsePoints()

    def calculateEllipsePoints(self):
        """
        Will produce points for an circle centemu at center, with orthonormal vectors a and b defining basis coordinates for the ellipse. c and d scale the respective axes.
        """
        t = np.linspace(0, 2 * np.pi * self.rotations, self.num_frames, endpoint=False)
        self.x = list(
            self.center[0]
            + self.c * np.cos(t) * self.a[0]
            + self.d * np.sin(t) * self.b[0]
        )
        self.y = list(
            self.center[1]
            + self.c * np.cos(t) * self.a[1]
            + self.d * np.sin(t) * self.b[1]
        )


class DoubleEllipse(Trajectory):
    def __init__(
        self,
        mu_a,
        mu_b,
        sigma_a,
        sigma_b,
        num_frames,
        dim,
        radius,
        center,
        a,
        b,
        c,
        d,
        rotations,
    ):
        super().__init__(mu_a, mu_b, sigma_a, sigma_b, num_frames, dim)
        self.radius = radius
        self.center = center
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.rotations = rotations
        self.n_ellipses = len(radius)

        self.calculateManyEllipsePoints()

    def calculateManyEllipsePoints(self):
        t = np.linspace(0, 2 * np.pi * self.rotations, self.num_frames, endpoint=False)
        self.x = []
        self.y = []
        sign = [1.0, -1.0]
        for i in range(self.n_ellipses):
            self.x.append(
                list(
                    self.center[i, 0]
                    + self.c[i] * np.cos(t * sign[i]) * self.a[i, 0]
                    + self.d[i] * np.sin(t * sign[i]) * self.b[i, 0]
                )
            )
            self.y.append(
                list(
                    self.center[i, 1]
                    + self.c[i] * np.cos(t * sign[i]) * self.a[i, 1]
                    + self.d[i] * np.sin(t * sign[i]) * self.b[i, 1]
                )
            )

    def getValuesAtT(self, i):
        if i < 1:
            mu = np.array(
                [
                    [
                        [self.x[0][0], self.x[1][0]],
                        [self.y[0][0], self.y[1][0]],
                    ],
                    [
                        [self.x[0][0], self.x[1][0]],
                        [self.y[0][0], self.y[1][0]],
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        [self.sigma_xt[0][0], self.sigma_xt[0][1]],
                        [self.sigma_yt[0][0], self.sigma_yt[0][1]],
                    ],
                    [
                        [self.sigma_xt[0][0], self.sigma_xt[0][1]],
                        [self.sigma_yt[0][0], self.sigma_yt[0][1]],
                    ],
                ]
            )
        else:
            mu = np.array(
                [
                    [
                        [self.x[0][i], self.x[1][i]],
                        [self.y[0][i], self.y[1][i]],
                    ],
                    [
                        [self.x[0][i - 1], self.x[1][i - 1]],
                        [self.y[0][i - 1], self.y[1][i - 1]],
                    ],
                ]
            )
            sigma = np.array(
                [
                    [
                        [self.sigma_xt[i][0], self.sigma_xt[i][1]],
                        [self.sigma_yt[i][0], self.sigma_yt[i][1]],
                    ],
                    [
                        [self.sigma_xt[i - 1][0], self.sigma_xt[i - 1][1]],
                        [self.sigma_yt[i - 1][0], self.sigma_yt[i - 1][1]],
                    ],
                ]
            )
        return mu, sigma


def trajectory_generator(mu_a, mu_b, sigma_a, sigma_b, num_frames, type):
    """
    Create density trajectories. Currently supports diagonal and circle
    """

    def ellipse_points(radius, num_frames, center, a, b, c, d, rotations):
        """
        Will produce points for an circle centemu at center, with orthonormal vectors a and b defining basis coordinates for the ellipse. c and d scale the respective axes.
        """
        t = np.linspace(0, 2 * np.pi * rotations, num_frames, endpoint=False)
        x = center[0] + c * np.cos(t) * a[0] + d * np.sin(t) * b[0]
        y = center[1] + c * np.cos(t) * a[1] + d * np.sin(t) * b[1]

        return x, y

    if type == "diagonal":

        def mu_xt_generator(t):
            return mu_a[0] * (1 - float(t) / num_frames) + mu_b[0] * (
                float(t) / num_frames
            )

        def mu_yt_generator(t):
            return mu_a[1] * (1 - float(t) / num_frames) + mu_b[1] * (
                float(t) / num_frames
            )

        def sigma_xt_generator(t):
            return sigma_a[0] * (1 - float(t) / num_frames) + sigma_b[0] * (
                float(t) / num_frames
            )

        def sigma_yt_generator(t):
            return sigma_a[1] * (1 - float(t) / num_frames) + sigma_b[1] * (
                float(t) / num_frames
            )

        mu_xt = [mu_xt_generator(k) for k in range(num_frames)]
        mu_yt = [mu_yt_generator(k) for k in range(num_frames)]
        sigma_xt = [sigma_xt_generator(k) for k in range(num_frames)]
        sigma_yt = [sigma_yt_generator(k) for k in range(num_frames)]
    elif type == "ellipse":
        radius = 5
        center = np.array([0, 0, 0])
        rotations = 10

        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = 1
        d = 1

        [mu_xt_generator, mu_yt_generator] = ellipse_points(
            radius, num_frames, center, a, b, c, d, rotations
        )

        def sigma_xt_generator(t):
            return sigma_a[0] * (1 - float(t) / num_frames) + sigma_b[0] * (
                float(t) / num_frames
            )

        def sigma_yt_generator(t):
            return sigma_a[1] * (1 - float(t) / num_frames) + sigma_b[1] * (
                float(t) / num_frames
            )

        mu_xt = list(mu_xt_generator)
        mu_yt = list(mu_yt_generator)
        sigma_xt = [sigma_xt_generator(k) for k in range(num_frames)]
        sigma_yt = [sigma_yt_generator(k) for k in range(num_frames)]
    else:
        raise ValueError("not a valid trajectory type")

    mu_params = {
        "mu_xt": mu_xt,
        "mu_yt": mu_yt,
        "sigma_xt": sigma_xt,
        "sigma_yt": sigma_yt,
    }
    return mu_params
