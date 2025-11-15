# Copyright (C) 2025 Brandon Bao
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
Python file containing utilities for basic Voronoi operations in 2D
"""

import numpy as np
from scipy.spatial.qhull import Delaunay
from adaptive_quadrature import integrate_adaptive


__author__ = "Brandon Bao"
__version__ = "1.0"
__maintainer__ = "Brandon Bao"


def random_points_stratified(chull, num_points, random_seed=None):
    """
    Generate random points in convex hull using stratified sampling. Delaunay tetras will be created from the vertices, and will be uniformly randomly sampled based on the weighted area of the tetra
    """
    hull_vertices = np.array(chull.points)
    dim = int(np.size(chull.points[0]))
    triangulation = Delaunay(hull_vertices)

    # # Plot Simplices
    # def plot_tri(ax, points, tri):
    #     """
    #     https://stackoverflow.com/questions/20025784/how-to-visualize-2D-delaunay-triangulation-in-python
    #     """
    #     edges = collect_edges(tri)
    #     x = np.array([])
    #     y = np.array([])
    #     for (i,j) in edges:
    #         x = np.append(x, [points[i, 0], points[j, 0], np.nan])
    #         y = np.append(y, [points[i, 1], points[j, 1], np.nan])

    #     ax.plot(x, y, color='k', lw='0.5')

    #     ax.scatter(points[:,0], points[:,1], color='b')

    # def collect_edges(tri):
    #     edges = set()

    #     def sorted_tuple(a,b):
    #         return (a,b) if a < b else (b,a)
    #     # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    #     for (i0, i1, i2) in tri.simplices:
    #         edges.add(sorted_tuple(i0,i1))
    #         edges.add(sorted_tuple(i0,i2))
    #         edges.add(sorted_tuple(i1,i2))
    #     return edges

    # fig = plt.figure()
    # ax = plt.axes()
    # plot_tri(ax, hull_vertices, triangulation)
    # plt.show()

    simplices = triangulation.simplices
    points = triangulation.points
    vertices = points[simplices, :]
    # print(vertices)
    # plt.triplot(points[:,0], points[:,1], simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()

    # shoelace method for area or area
    area = [0.5 * abs(np.linalg.det(np.c_[v, np.ones(int(dim + 1))])) for v in vertices]
    totalv = sum(area)
    weights = area / totalv
    decimals = int(np.log10(num_points))
    # Want exactly num_points so need to minimally modify expected weights that are used to determine the number of points needed

    def round_series_retain_integer_sum(xs):
        """
        https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal
        """
        N = sum(xs)
        Rs = [int(x) for x in xs]
        K = N - sum(Rs)
        K = round(K)
        # assert K == int(K)
        fs = [x - int(x) for x in xs]
        indices = [
            i
            for order, (e, i) in enumerate(
                reversed(sorted((e, i) for i, e in enumerate(fs)))
            )
            if order < K
        ]
        ys = [R + 1 if i in indices else R for i, R in enumerate(Rs)]
        return ys

    counts_full = np.array(round_series_retain_integer_sum(num_points * weights))
    # counts_full = (num_points*rounded_weights).astype(int)
    # counts_full = (float(num_points)*rounded_weights)
    # print(math.fsum(counts_full))
    # counts_full = [int(round(x)) for x in counts_full]
    # print(math.fsum(counts_full))
    sampled_points = []
    num_sampled = []
    sampled_points_append = sampled_points.append
    num_sampled_append = num_sampled.append
    for idx, simplex in enumerate(triangulation.simplices):
        if counts_full[idx] == 0:
            num_sampled_append(0)
            continue
        # r, s, t, u = -np.log(np.random.uniform(size=counts_full[idx])), -np.log(np.random.uniform(size=counts_full[idx])), -np.log(
        #     np.random.uniform(size=counts_full[idx])), -np.log(np.random.uniform(size=counts_full[idx]))
        # var_sum = r + s + t + u

        test_samples = -np.log(np.random.uniform(size=(counts_full[idx], int(dim + 1))))
        test_sum = np.sum(test_samples, axis=1)
        test_points = test_samples / test_sum[:, np.newaxis]

        # xi_a, xi_b, xi_c, xi_d = (r/var_sum)[:, None], (s/var_sum)[:, None], (t/var_sum)[
        #     :, None], (u/var_sum)[:, None]  # vector of length num_points
        simplex_vertices = points[simplex, :]
        simplex_points = np.dot(test_points, simplex_vertices)

        # simplex_points = xi_a * simplex_vertices[0, :] + xi_b * simplex_vertices[1,
        #                                                                          :] + xi_c * simplex_vertices[2, :] + xi_d * simplex_vertices[3, :]

        # Plot points in individual simplex
        # fig = plt.figure()
        # ax = plt.axes(projection='2D')
        # ax.scatter(simplex_points[:,0], simplex_points[:,1], simplex_points[:,2], color='r',s=1)
        # plt.show()
        num_sampled_append(len(simplex_points))
        sampled_points_append(simplex_points)

    # used in chi_square_stratified function
    num_sampled = np.vstack(num_sampled)
    sampled_points = np.vstack((sampled_points))

    # fig = plt.figure()
    # ax = plt.axes(projection='2D')
    # ax.scatter(sampled_points[:,0], sampled_points[:,1], sampled_points[:,2], color='r',s=1)
    # plt.show()

    return sampled_points, num_sampled.flatten()


def get_phi(model, tau=5):
    def phi(x, y, t, tau=tau):
        """
        Density functions for robotarium
        """
        if model == 1:
            f = np.exp(-1 * ((x - 2 * np.sin(t / tau)) ** 2 + (y) ** 2))
        elif model == 2:
            f = np.exp(
                -1 * ((x - np.sin(t / tau)) ** 2 + (y + np.sin(2 * t / tau)) ** 2)
            )  # it moves faster
        elif model == 3:
            f = np.exp(
                -1 * ((x - 2 * np.cos(t / tau)) ** 2 + (y - 2 * np.sin(t / tau)) ** 2)
            )
        elif model == 4:
            # if not (isinstance(sigma, float) or isinstance(sigma, int)):
            # if sigma.size == 2:
            # Phi 4
            tau = 1500 / (2 * np.pi)  # 20 # L: tau
            r = 0.3  # 5
            s1 = 1  # 1
            s2 = 1
            offset = 1.6 / 3.0
            f1 = np.exp(
                -1
                * (
                    ((x - r * np.cos(t / tau) + offset) / s1) ** 2
                    + ((y - r * np.sin(t / tau)) / s1) ** 2
                )
            )
            f2 = np.exp(
                -1
                * (
                    ((x - r * np.cos(-t / tau) - offset) / s2) ** 2
                    + ((y - r * np.sin(-t / tau)) / s2) ** 2
                )
            )
            f = f1 + f2
        else:
            raise ValueError("Model should be 1-4")
        return f

    return phi
    # # Phi 6 Density Function
    # tau = 1000 / (2 * np.pi)  # 20 # L: tau
    # r = 0.6  # 5, 0.6 for robotatrium
    # s = sigma  # 1
    # f = np.exp(
    #     -1
    #     * (((x - r * np.cos(t / tau)) / s) ** 2 + ((y - r * np.sin(t / tau)) / s) ** 2)
    # )

    # return f


def get_phidot(model=1, tau=5):
    """
    Used to calculate dc/dt
    """
    phi = get_phi(model)

    def phidot(x, y, t, tau=tau):
        if model == 1:
            f = phi(
                x,
                y,
                t,
                tau=tau,
            ) * (4 / tau * (x - 2 * np.sin(t / tau)) * np.cos(t / tau))
        elif model == 2:
            f = phi(
                x,
                y,
                t,
                tau=tau,
            ) * (
                2 / tau * (x - np.sin(t / tau)) * np.cos(t / tau)
                - 4 / tau * (y + np.sin(2 * t / tau)) * np.cos(2 * t / tau)
            )
        elif model == 3:
            f = phi(
                x,
                y,
                t,
                tau=tau,
            ) * (
                -4 / tau * (x - 2 * np.cos(t / tau)) * np.sin(t / tau)
                + 4 / tau * (y - 2 * np.sin(t / tau)) * np.cos(t / tau)
            )
        elif model == 4:

            # Phi 5
            tau = 1500 / (2 * np.pi)  # 20 # L: tau
            r = 0.3  # 5
            s1 = 1  # 1
            s2 = 1
            offset = 1.6 / 3.0
            f1 = np.exp(
                -1
                * (
                    ((x - r * np.cos(t / tau) + offset) / s1) ** 2
                    + ((y - r * np.sin(t / tau)) / s1) ** 2
                )
            )
            f2 = np.exp(
                -1
                * (
                    ((x - r * np.cos(-t / tau) - offset) / s2) ** 2
                    + ((y - r * np.sin(-t / tau)) / s2) ** 2
                )
            )
            f = (
                f1
                * (
                    -2
                    * r
                    / (tau * s1**2)
                    * (x - r * np.cos(t / tau) + offset)
                    * -1.0
                    * np.sin(t / tau)
                    + 2
                    * r
                    / (tau * s1**2)
                    * (y - r * np.sin(t / tau))
                    * -1.0
                    * np.cos(t / tau)
                )
            ) + (
                f2
                * (
                    -2
                    * r
                    / (tau * s2**2)
                    * (x - r * np.cos(t / tau) - offset)
                    * -1.0
                    * np.sin(-t / tau)
                    + 2
                    * r
                    / (tau * s2**2)
                    * (y - r * np.sin(t / tau))
                    * -1.0
                    * np.cos(-t / tau)
                )
            )
            f = f1 + f2
        else:
            raise ValueError("Model should be 1-4")
        return f

    return phidot
    # # Phi 6 Density Function
    # tau = 1000 / (2 * np.pi)  # L: tau
    # r = 0.6  # 5, 0.6 for robotatrium
    # s = sigma
    # f = phi(x, y, mu, sigma, t) * (
    #     -2 * r / (tau * s**2) * (x - r * np.cos(t / tau)) * np.sin(t / tau)
    #     + 2 * r / (tau * s**2) * (y - r * np.sin(t / tau)) * np.cos(t / tau)
    # )
    # return f


def calculate_mass_and_centroid(poly, phi, t, pi_vec, eps_rel=1e-6):
    # Check that the generating point is inside the polygon
    m_vec = []
    c_vec = []
    for i, vor in enumerate(poly):
        vertices = np.array(vor["vertices"])
        pi = pi_vec[i]
        angles = [
            (np.arctan2(vertices[i][1] - pi[1], vertices[i][0] - pi[0]), vertices[i])
            for i in range(len(vertices))
        ]
        angles.sort()
        vertices = np.array([angles[k][1] for k in range(len(angles))])
        # try:
        #     sh_pol = Polygon(vertices)
        #     sh_pi = Point(pi)
        #     if not sh_pol.contains(sh_pi):
        #         print('Polygon doesnt contain: ', pi)
        # except:
        #     if poly:
        #         print('the vertex may not form a polygon')
        #     else:
        #         print('Polygon is empty')

        def phi2(q):
            return phi(q[0], q[1], t)

        def psi(q):
            return q * phi(q[0], q[1], t)

        # Calculate centroid and mass
        m = 0

        c = np.zeros([2])
        for k in range(1, len(vertices) - 1):
            triangle = [
                list(vertices[0]),
                list(vertices[k]),
                list(vertices[k + 1]),
            ]
            [mk, m_error] = integrate_adaptive(
                phi2, triangle, eps_rel, minimum_triangle_area=0
            )
            [ck, c_error] = integrate_adaptive(
                psi, triangle, eps_rel, minimum_triangle_area=0
            )

            # sh_triangle = Polygon(triangle)
            # sh_c = Point(ck / mk)
            # if not sh_triangle.contains(sh_c):
            #     print('Triangle integration error: ', k)
            #     print(triangle)
            #     print(ck)
            m += mk
            c += np.array(ck)

        c = c / m
        m_vec.append(m)
        c_vec.append(c)

    return np.array(m_vec), np.array(c_vec)


def calculate_dcdt(poly, dphidt, t, m_vec, c_vec, pi_vec, eps_rel=1e-6):
    dcdt_vec = []
    for i, vor in enumerate(poly):
        vertices = np.array(vor["vertices"])
        pi = pi_vec[i]
        angles = [
            (np.arctan2(vertices[i][1] - pi[1], vertices[i][0] - pi[0]), vertices[i])
            for i in range(len(vertices))
        ]
        angles.sort()
        vertices = np.array([angles[k][1] for k in range(len(angles))])
        dcdt = np.zeros([1, 2])
        # Calculate time derivative
        m = m_vec[i]
        c = c_vec[i]

        def psi2(q):
            return np.array([q[0] - c[0], q[1] - c[1]]) * dphidt(q[0], q[1], t) / m

        for k in range(1, len(vertices) - 1):
            triangle = [
                list(vertices[0]),
                list(vertices[k]),
                list(vertices[k + 1]),
            ]
            [dck, dc_error] = integrate_adaptive(
                psi2, triangle, eps_rel, minimum_triangle_area=0
            )
            dcdt += np.array(dck)
        dcdt_vec.append(dcdt)

    return np.array(dcdt_vec)


def calculate_cost(poly, phi, t, pi_vec, eps_rel=1e-6):
    J_vec = []
    for i, vor in enumerate(poly):
        vertices = np.array(vor["vertices"])
        pi = pi_vec[i]
        angles = [
            (np.arctan2(vertices[i][1] - pi[1], vertices[i][0] - pi[0]), vertices[i])
            for i in range(len(vertices))
        ]
        angles.sort()
        vertices = np.array([angles[k][1] for k in range(len(angles))])
        J = 0
        pi = pi_vec[i]

        def H(q):
            try:
                qq = np.sqrt((q[0] - pi[0]) ** 2 + (q[1] - pi[1]) ** 2)
            except:
                print("Error in Cost Function")
            return qq * phi(q[0], q[1], t)

        for k in range(1, len(vertices) - 1):
            triangle = [
                list(vertices[0]),
                list(vertices[k]),
                list(vertices[k + 1]),
            ]
            Jk = 0
            [Jk, J_error] = integrate_adaptive(
                H, triangle, eps_rel, minimum_triangle_area=0
            )
            J += Jk
        J_vec.append(J)
    return sum(J_vec)


# def weighted_phi(
#     x, y, mu, sigma, t, dt=None, c=None, pi=None, sig_dig=4, model=1, tau=5
# ):
#     """
#     Weighted Gaussian used to find the centroid
#     """
#     # return np.around(
#     #     np.transpose(np.multiply(np.array([x, y, z]) , phi(x,y,z,mu,sigma,sig_dig=4))), sig_dig
#     # )
#     return np.multiply(
#         np.array([x, y]), phi(x, y, mu, sigma, t, sig_dig=4, model=model, tau=tau)
#     ).T


# def dcdt_function(x, y, mu, sigma, t, dt, c, pi=None, sig_dig=4, model=1, tau=5):
#     """
#     Calculate dc/dt for Time Varying Densities
#     """
#     # return np.around(np.multiply((np.array([x, y, z]).T - c).T, phidot(x,y,np.array([mu[0]]),np.array([sigma[0]]),np.array([mu[1]]),np.array([sigma[1]]),dt,sig_dig=sig_dig)), sig_dig).T
#     return np.multiply(
#         (np.array([x, y]).T - c).T,
#         phidot(
#             x,
#             y,
#             np.array([mu[0]]),
#             sigma,
#             np.array([mu[1]]),
#             sigma,  # TODO keep sigma as size 2, so that the phidot can be appoximated
#             t,
#             dt,
#             sig_dig=sig_dig,
#             model=model,
#             tau=tau,
#         ),
#     ).T


# def cost_function(x, y, mu, sigma, t, dt=1, c=None, pi=None, sig_dig=4, model=1, tau=5):
#     """
#     Calculate locational cost function H to compare different coverage control algorithms
#     """
#     try:
#         qq = np.sqrt((x - pi[0]) ** 2 + (y - pi[1]) ** 2)
#     except ValueError:
#         print("Oops!  That was no valid number.  Try again...")
#     return qq * phi(x, y, mu, sigma, t, sig_dig=sig_dig, model=model, tau=tau) * dt


def monte_carlo_integrate(
    points_all,
    area,
    function,
    mu,
    sigma,
    t,
    dt,
    c,
    pos,
    num_points=None,
    model=1,
    tau=5,
):
    """
    Used by lloyd_single_step to find mass and centroid using sampled points. Takes in sampled points, a function such as phi/weighted_phi, and the parameters of the mu location. The area normalizes the calculation
    """
    integral = []
    # dim = np.shape(points_all)[2]
    # if dim == 2: # add z points so that function eval works
    #     points_all = [np.c_[points, np.zeros(shape=(len(points),))] for points in points_all]
    # Go one voronoi partition at a time
    for idx, points in enumerate(points_all):
        # Evaluate function on points
        function_result = function(
            points[:, 0],
            points[:, 1],
            mu,
            sigma,
            t,
            dt,
            c[idx],
            pos[idx],
            sig_dig=4,
            model=model,
            tau=tau,
        )
        # Plot Monte Carlo Function Evaluation Result
        # visualize_monte_carlo(points,sigma,function_result)
        integral_result = np.multiply(
            area[idx] / num_points, np.sum(function_result, axis=0)
        )
        integral.append(integral_result)

    return np.array(integral)  # in same order as inputted through vor


# def massless_centroid(pos, mu_pos, vor, partitions):
#     """
#     Find centroid when mass is zero

#     The centroid must remain in the Voronoi cell and we want it to be as close as possible to the red position
#     We take a ray which is from the agent position to the mu position and test if it intersects a face
#     Then we find the point where it intersects.
#     TODO: Make sure it works in edge cases
#     """
#     # print("Line 679 Position after Massless: ", pos)
#     dim = 2
#     # if pos[-1] == 0.:
#     #     dim = 2
#     vertices = vor["vertices"]
#     if check_membership(vertices, mu_pos):
#         centroid = mu_pos
#         return centroid
#     norm_dist = np.linalg.norm(mu_pos - pos)
#     rayDirection = (mu_pos - pos) / norm_dist
#     rayPoint = pos
#     faces = vor["faces"]
#     centroid = [0, 0, 0]
#     centroids = []
#     prevdist = norm_dist
#     for face in faces:
#         verticesIdx = face["vertices"]
#         A = vertices[verticesIdx[0]]
#         B = vertices[verticesIdx[1]]
#         # C = vertices[verticesIdx[2]]
#         # planeNormal = np.cross(B-A,C-A)
#         # t = (planeNormal[0]*(A[0]-pos[0]) + planeNormal[1]*(A[1]-pos[1]) + planeNormal[2]*(A[2]-pos[2]))/(np.dot(A,A))
#         # proj_pi = np.array([pos[0]+t*planeNormal[0],pos[1]+t*planeNormal[1],pos[2]+t*planeNormal[2]])
#         # orientation = np.dot(proj_pi-pos,planeNormal)
#         # if orientation < 0:
#         #     planeNormal *= -1
#         # planePoint = A
#         if dim == 2:
#             intersect = point_line_intersect(A[0:2], B[0:2], mu_pos, pos)
#             # print(intersect)
#             if len(intersect) != 0:
#                 # intersect =  np.r_[intersect, 0.]
#                 dist = np.linalg.norm(mu_pos - intersect)
#             else:
#                 dist = prevdist
#         # else:
#         #     intersect = point_plane_intersect(planeNormal,planePoint,rayDirection,rayPoint)
#         #     if intersect.size:
#         #         dist = np.linalg.norm(mu_pos-intersect)
#         #         # print(intersect, intersect.size)
#         #     else:
#         #         print("Edge case, unexpected Voronoi cell configuration")
#         # check if there exists a valid intersection
#         centroids.append(intersect)
#         # print(dist,prevdist)
#         if dist < prevdist:
#             centroid = intersect
#             prevdist = dist

#     # fig = plt.figure()
#     # ax = fig.add_subplot(111,projection='2D')
#     # ax.scatter(centroid[0], centroid[1] , centroid[2],  color='green')
#     # ax.scatter(pos[0], pos[1] , pos[2],  color='blue')
#     # ax.scatter(mu_pos[0], mu_pos[1] , mu_pos[2],  color='red')

#     # generate_plots.plot_voronoi_regions(pos,partitions,ax=ax)
#     # plt.show()
#     # print(centroid)
#     return centroid
