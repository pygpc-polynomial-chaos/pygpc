from __future__ import division


import numpy as np

from pygpc.AbstractModel import AbstractModel

class SphereModel(AbstractModel):

#    def __init__(self, conductivities, context, radii, anode_pos, cathode_pos, p, nbr_polynomials=50):
    def __init__(self, parameters, context):
        super(SphereModel, self).__init__(parameters, context)

        self.nbr_polynomials = 50

    # copied from 'sphere.py' -> potential_3layers_surface_electrodes
    def simulate( self, process_id):
        """Calculates the electric potential in a 3-layered sphere caused by point-like electrodees


        Parameters
        -------------------------------
        conductivities: list of lenght = 3
            Conductivity of the 3 layers (innermost to outermost), in S/m
        radii: list of length = 3
            Radius of each of the 3 layers (innermost to outermost), in mm
        anode_pos: 3x1 ndarray
            position of the anode_pos, in mm
        cathode_pos: 3x1 ndarray
            position of cathode_pos, in mm
        p: (Nx3)ndarray
            List of positions where the poteitial should be calculated, in mm
        nbr_polynomials: int
            Number of of legendre polynomials to use

        Returns
        ------------------------------
        (Nx1) ndarray
            Values of the electric potential, in V

        Reference
        ------------------------------
        S.Rush, D.Driscol EEG electrode sensitivity--an application of reciprocity.
        """
        assert len(self.parameters["R"]) == 3
        assert self.parameters["R"][0] < self.parameters["R"][1] and self.parameters["R"][1] < self.parameters["R"][2]
        #assert len(self.conductivities) == 3
        assert len(self.parameters["anode_pos"]) == 3
        assert len(self.parameters["cathode_pos"]) == 3
        assert self.parameters["points"].shape[1] == 3

        b_over_s = float(self.parameters["sigma_1"]) / float(self.parameters["sigma_2"])
        s_over_t = float(self.parameters["sigma_2"]) / float(self.parameters["sigma_3"])
        radius_brain = self.parameters["R"][0] * 1e-3
        radius_skull = self.parameters["R"][1] * 1e-3
        radius_skin = self.parameters["R"][2] * 1e-3

        r = np.linalg.norm(self.parameters["points"], axis=1) * 1e-3
        theta = np.arccos(self.parameters["points"][:, 2] * 1e-3 / r)
        phi = np.arctan2(self.parameters["points"][:, 1], self.parameters["points"][:, 0])

        p_r = np.vstack((r, theta, phi)).T

        cathode_pos = (np.sqrt(self.parameters["cathode_pos"][0]**2 + self.parameters["cathode_pos"][1]**2 + self.parameters["cathode_pos"][2]**2) * 1e-3,
                       np.arccos(self.parameters["cathode_pos"][2] /
                                 np.sqrt(self.parameters["cathode_pos"][0]**2 + self.parameters["cathode_pos"][1]**2 + self.parameters["cathode_pos"][2]**2)),
                       np.arctan2(self.parameters["cathode_pos"][1], self.parameters["cathode_pos"][0]))

        anode_pos = (np.sqrt(self.parameters["anode_pos"][0]**2 + self.parameters["anode_pos"][1]**2 + self.parameters["anode_pos"][2]**2) * 1e-3,
                     np.arccos(self.parameters["anode_pos"][2] /
                               np.sqrt(self.parameters["anode_pos"][0] ** 2 + self.parameters["anode_pos"][1]**2 + self.parameters["anode_pos"][2]**2)),
                     np.arctan2(self.parameters["anode_pos"][1], self.parameters["anode_pos"][0]))

        A = lambda n: ((2 * n + 1)**3 / (2 * n)) / (((b_over_s + 1) * n + 1) * ((s_over_t + 1) * n + 1) +
                                                    (b_over_s - 1) * (s_over_t - 1) * n * (n + 1) * (radius_brain / radius_skull)**(2 * n + 1) +
                                                    (s_over_t - 1) * (n + 1) * ((b_over_s + 1) * n + 1) * (radius_skull / radius_skin)**(2 * n + 1) +
                                                    (b_over_s - 1) * (n + 1) * ((s_over_t + 1) * (n + 1) - 1) * (radius_brain / radius_skin)**(2 * n + 1))
        # All of the bellow are modified: division by raidus_skin moved to the
        # coefficients calculations due to numerical constraints
        # THIS IS DIFFERENT FROM THE PAPER (there's a sum instead of difference)
        S = lambda n: (A(n)) * ((1 + b_over_s) * n + 1) / (2 * n + 1)
        U = lambda n: (A(n) * radius_skin) * n * (1 - b_over_s) * \
            radius_brain**(2 * n + 1) / (2 * n + 1)
        T = lambda n: (A(n) / ((2 * n + 1)**2)) *\
            (((1 + b_over_s) * n + 1) * ((1 + s_over_t) * n + 1) +
             n * (n + 1) * (1 - b_over_s) * (1 - s_over_t) * (radius_brain / radius_skull)**(2 * n + 1))
        W = lambda n: ((n * A(n) * radius_skin) / ((2 * n + 1)**2)) *\
            ((1 - s_over_t) * ((1 + b_over_s) * n + 1) * radius_skull**(2 * n + 1) +
             (1 - b_over_s) * ((1 + s_over_t) * n + s_over_t) * radius_brain**(2 * n + 1))

        brain_region = np.where(p_r[:, 0] <= radius_brain)[0]
        skull_region = np.where(
            (p_r[:, 0] > radius_brain) * (p_r[:, 0] <= radius_skull))[0]
        skin_region = np.where((p_r[:, 0] > radius_skull)
                               * (p_r[:, 0] <= radius_skin))[0]
        inside_sphere = np.where((p_r[:, 0] <= radius_skin))[0]
        outside_sphere = np.where((p_r[:, 0] > radius_skin))[0]

        cos_theta_a = np.cos(cathode_pos[1]) * np.cos(p_r[:, 1]) +\
            np.sin(cathode_pos[1]) * np.sin(p_r[:, 1]) * \
            np.cos(p_r[:, 2] - cathode_pos[2])
        cos_theta_b = np.cos(anode_pos[1]) * np.cos(p_r[:, 1]) +\
            np.sin(anode_pos[1]) * np.sin(p_r[:, 1]) * \
            np.cos(p_r[:, 2] - anode_pos[2])

        potentials = np.zeros((self.parameters["points"].shape[0]), dtype='float64')

        coefficients = np.zeros((self.nbr_polynomials, self.parameters["points"].shape[0]), dtype='float64')

        # accelerate
        for ii in range(1, self.nbr_polynomials):
            n = float(ii)
            coefficients[ii, brain_region] = np.nan_to_num(
                A(n) * ((p_r[brain_region, 0] / radius_skin)**n))

            coefficients[ii, skull_region] = np.nan_to_num(S(n) * (p_r[skull_region, 0] / radius_skin)**n +
                                                           U(n) * (p_r[skull_region, 0] * radius_skin)**(-n - 1))

            coefficients[ii, skin_region] = np.nan_to_num(T(n) * (p_r[skin_region, 0] / radius_skin)**n
                                                          + W(n) * (p_r[skin_region, 0] * radius_skin)**(-n - 1))

        potentials[inside_sphere] = np.nan_to_num(
            np.polynomial.legendre.legval(cos_theta_a[inside_sphere], coefficients[:, inside_sphere], tensor=False) -
            np.polynomial.legendre.legval(cos_theta_b[inside_sphere], coefficients[:, inside_sphere], tensor=False))

        potentials *= 1.0 / (2 * np.pi * self.parameters["sigma_3"] * radius_skin)

        potentials[outside_sphere] = 0.0

        return potentials
        # plot_scatter(points_cart[:,0],points_cart[:,1],points_cart[:,2],potentials)