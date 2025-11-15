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

# CODE ADAPTED FROM QUADPY
# the original code imposes accuracy with absolute errors, the modified code imposes accuracy with relative error
import numpy as np
from quadpy import t2

# from quadpy.nsimplex import get_vol


def _numpy_all_except(a, axis=-1):
    axes = np.arange(a.ndim)
    axes = np.delete(axes, axis)
    return np.all(a, axis=tuple(axes))


def integrate_adaptive(
    f,
    triangles,
    eps_rel,
    minimum_triangle_area=None,
    scheme1=t2.get_good_scheme(10),
    scheme2=t2.get_good_scheme(10),
    # scheme1=t2.dunavant_05(),
    # scheme2=t2.dunavant_10(),
    dot=np.dot,
):
    sumfun = np.sum

    triangles = np.array(triangles)
    if len(triangles.shape) == 2:
        # add dimension in the second-to-last place
        triangles = np.expand_dims(triangles, -2)

    areas = t2.get_vol(triangles)
    total_area = sumfun(areas)

    if minimum_triangle_area is None:
        minimum_triangle_area = 0

    val1 = scheme1.integrate(f, triangles, dot=dot)
    val2 = scheme2.integrate(f, triangles, dot=dot)
    error_estimate = abs(val1 - val2)
    total_val = sumfun(val2, axis=-1)
    total_val = abs(total_val)

    # Mark intervals with acceptable approximations. For this, take all()
    # across every dimension except the last one, which is the interval index.
    # is_good = _numpy_all_except(error_estimate < eps_rel * areas / total_area, axis=-1)

    # CAREFUL WITH TOTAL_VAL = 0

    is_good = np.zeros(np.shape(error_estimate)[-1], dtype="bool")
    for i in range(np.shape(error_estimate)[-1]):
        if np.all(
            np.logical_or(
                error_estimate[..., i] < total_val * eps_rel * areas[i] / total_area,
                total_val == 0,
            )
        ):
            is_good[i] = True

    # add values from good intervals to sum
    quad_sum = sumfun(val1[..., is_good], axis=-1)
    global_error_estimate = sumfun(error_estimate[..., is_good], axis=-1)

    is_bad = np.logical_not(is_good)
    while any(is_bad):
        # split the bad triangles into four #triforce
        #
        #         /\
        #        /__\
        #       /\  /\
        #      /__\/__\
        #
        triangles = triangles[..., is_bad, :]
        midpoints = [
            0.5 * (triangles[1] + triangles[2]),
            0.5 * (triangles[2] + triangles[0]),
            0.5 * (triangles[0] + triangles[1]),
        ]
        triangles = np.array(
            [
                np.concatenate(
                    [triangles[0], triangles[1], triangles[2], midpoints[0]]
                ),
                np.concatenate(
                    [midpoints[1], midpoints[2], midpoints[0], midpoints[1]]
                ),
                np.concatenate(
                    [midpoints[2], midpoints[0], midpoints[1], midpoints[2]]
                ),
            ]
        )
        areas = t2.get_vol(triangles)

        assert all(areas > minimum_triangle_area)

        # compute values and error estimates for the new intervals
        val1 = scheme1.integrate(f, triangles, dot=dot)
        val2 = scheme2.integrate(f, triangles, dot=dot)
        error_estimate = abs(val1 - val2)
        total_val = quad_sum + sumfun(val2, axis=-1)
        total_val = abs(total_val)

        # mark good intervals, gather values and error estimates
        # is_good = _numpy_all_except(error_estimate < eps_rel * areas / total_area, axis=-1)
        is_good = np.zeros(np.shape(error_estimate)[-1], dtype="bool")
        for i in range(np.shape(error_estimate)[-1]):
            if np.all(
                error_estimate[..., i] < total_val * eps_rel * areas[i] / total_area
            ):
                is_good[i] = True

        # add values from good intervals to sum
        quad_sum += sumfun(val1[..., is_good], axis=-1)
        global_error_estimate += sumfun(error_estimate[..., is_good], axis=-1)
        is_bad = np.logical_not(is_good)

    return quad_sum, global_error_estimate


# MY SHITTY CODE
# is_good = numpy.zeros(numpy.shape(triangles)[1], dtype=bool)
# val1 = []
# val2 = []
# error_estimate = []
# for k in range(len(is_good)):
#     triangle = triangles[:,k]
#     val1k = scheme1.integrate(f, triangle, dot=dot)
#     val2k = scheme2.integrate(f, triangle, dot=dot)
#     error_estimatek = abs(val1k - val2k)
#     val1.append(val1k)
#     val2.append(val2k)
#     error_estimate.append(error_estimatek)
#     if areas[k] < minimum_triangle_area:
#         is_good[k] = True
#     if areas[k] > minimum_triangle_area:
#         is_good[k] = numpy.all(error_estimate < eps * areas[k] / total_area)
#
# val1 = numpy.array(val1)
# val2 = numpy.array(val2)
# error_estimate = numpy.array(error_estimate)

# add values from good intervals to sum
# quad_sum += sumfun(val1[..., is_good], axis=-1)
# global_error_estimate += sumfun(error_estimate[..., is_good], axis=-1)
# is_bad = numpy.logical_not(is_good)
