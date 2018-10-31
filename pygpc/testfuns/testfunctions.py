# -*- coding: utf-8 -*-
import numpy as np

from pygpc.AbstractModel import AbstractModel

class Peaks:

    def __init__(self, parameters, context):
        super(Peaks, self).__init__(parameters, context)

    def simulate(self, process_id):
        """ 2-dimensional peaks function.
    
            Input:
                 x       ... input data [N_input x 2]
            Output:
                res     ... result [N_input x 1]
        """

        y = np.array([(3.0*(1-self.parameters["x"][:,0])**2.*np.exp(-(self.parameters["x"][:,0]**2) - (self.parameters["x"][:,1]+1)**2) \
            - 10.0*(self.parameters["x"][:,0]/5.0 - self.parameters["x"][:,0]**3 - self.parameters["x"][:,1]**5)*np.exp(-self.parameters["x"][:,0]**2-self.parameters["x"][:,1]**2)\
            - 1.0/3*np.exp(-(self.parameters["x"][:,0]+1)**2 - self.parameters["x"][:,1]**2))]).transpose()
    
        return y

class Lim2002:
    def __init__(self, parameters, context):
        super(Lim2002, self).__init__(parameters, context)

    def simulate(self, process_id):
        """ 2-dimensional test function of Lim et al.
        This function is a polynomial in two dimensions, with terms up to degree
        5. It is nonlinear, and it is smooth despite being compleparameters["x"], which is
        common for computer experiment functions (Lim et al., 2002). 
        
        Lim, Y. B., Sacks, J., Studden, W. J., & Welch, W. J. (2002). Design
        and analysis of computer experiments when the output is highly correlated
        over the input space. Canadian Journal of Statistics, 30(1), 109-126.
        
        f(x) = 9 + 5/2*x1 - 35/2*x2 + 5/2*x1*x2 + 19*x2^2 - 15/2*x1^3 
               - 5/2*x1*x2^2 - 11/2*x2^4 + x1^3*x2^2         
        
        y = lim_2002(x)
        
        Input:  
                x ... input data [N_input x 2]
                      xi ∈ [0, 1], for all i = 1, 2
                
        Output: 
                y ... result [N_input x 1]
        """
    
        y = np.array([(9 + 5.0/2*self.parameters["x"][:,0] - 35.0/2*self.parameters["x"][:,1] + 5.0/2*self.parameters["x"][:,0]*self.parameters["x"][:,1]
        + 19*self.parameters["x"][:,1]**2 - 15.0/2*self.parameters["x"][:,0]**3 - 5.0/2*self.parameters["x"][:,0]*self.parameters["x"][:,1]**2 - 11.0/2*self.parameters["x"][:,1]**4
        + self.parameters["x"][:,0]**3*self.parameters["x"][:,1]**2)]).transpose()
    
        return y
    
class Ishigami:
    def __init__(self, parameters, context):
        super(Ishigami, self).__init__(parameters, context)

    def simulate(self, process_id):
        """ 3-dimensional test function of Ishigami.
        The Ishigami function of Ishigami & Homma (1990) is used as an example
        for uncertainty and sensitivity analysis methods, because it exhibits
        strong nonlinearity and nonmonotonicity. It also has a peculiar
        dependence on x3, as described by Sobol' & Levitan (1999).
        
        Ishigami, T., & Homma, T. (1990, December). An importance quantification
        technique in uncertainty analysis for computer models. In Uncertainty
        Modeling and Analysis, 1990. Proceedings., First International Symposium
        on (pp. 398-403). IEEE.
        
        Sobol', I. M., & Levitan, Y. L. (1999). On the use of variance reducing
        multipliers in Monte Carlo computations of a global sensitivity index.
        Computer Physics Communications, 117(1), 52-61.
        
        f(x) = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)        
        
        y = ishigami(x,a,b)
        
        Input:  
                x   ... input data [N_input x 3]
                        xi ~ Uniform[-π, π], for all i = 1, 2, 3
                a,b ... shape parameter
                
        Output: 
                y   ... result [N_input x 1]
         """
        y = np.array([(np.sin(self.parameters["x"][:,0])+self.parameters["a"]*np.sin(self.parameters["x"][:,1])**2+self.self.parameters["b"]*self.parameters["x"][:,2]**4*np.sin(self.parameters["x"][:,0]))]).transpose()
    
        return y
        
class Sphere0Fun:

    def __init__(self, parameters, context):
        super(Sphere0Fun, self).__init__(parameters, context)


    def simulate(self, process_id):
        """ N-dimensional sphere function with zero mean.
        
        y = sphere0(x,a,b)

        Input:  
                self.x   ... input data [N_input x N_dims]
                a,b ... lower and upper bound of all input vars
        Output: 
                y   ... result [N_input x 1]
        """

        try:
            N = self.parameters["x"].shape[1]
        except IndexError:
            N=1
            self.parameters["x"]=np.array([self.parameters["x"]])

        # zero mean
        c2 = (1.0*N*(self.parameters["b"]**3-self.parameters["a"]**3))/(3*(self.parameters["b"]-self.parameters["a"]))
    
        # sphere function
        y = np.array([(np.sum(np.square(self.parameters["x"]),axis=1)-c2)]).transpose()
    
        return y


class SphereFun:

    def __init__(self, parameters, context):
        super(SphereFun, self).__init__(parameters, context)


    def simulate(self, process_id):
        """ N-dimensional sphere function with zero mean.
        
        y = sphere0(x)
        
        Input:  
                x ... input data [N_input x N_dims]
        Output: 
                y ... result [N_input x 1]
        """
    
        # sphere function
        y = np.array([(np.sum(np.square(self.parameters["x"]),axis=1))]).transpose()
    
        return y


class GFunction:

    def __init__(self, parameters, context):
        super(GFunction, self).__init__(parameters, context)

    def simulate(self, process_id):
        """ N-dimensional g-function used by Saltelli and Sobol
        this test function is used as an integrand for various numerical 
        estimation methods, including sensitivity analysis methods, because it 
        is fairly complex, and its sensitivity indices can be expressed 
        analytically. The exact value of the integral with this function as an 
        integrand is 1. 
        
        Saltelli, Andrea; Sobol, I. M. (1995): Sensitivity analysis for nonlinear
        mathematical models: numerical experience. In: Mathematical models and
        computer experiment 7 (11), S. 16–28.
        
        y = g_function(x,a)
        
        Input:  
                x ... input data [N_input x N_dims]
                a ... importance factor of dimensions [N_dims]
        Output: 
                y ... result [N_input x 1]
        """
         
        try:
            self.parameters["x"].shape[1]
        except IndexError:
            self.parameters["x"]=np.array([self.parameters["x"]])

        # g-function
        y = np.array([(np.prod((np.abs(4.0*self.parameters["x"]-2)+self.parameters["a"])/(1.0+self.parameters["a"]),axis=1))]).transpose()
    
        return y
    
class OakleyOhagan2004:

    def __init__(self, parameters, context):
        super(OakleyOhagan2004, self).__init__(parameters, context)

    def simulate(self, process_id):
        """ 15-dimensional test function of OAKLEY & O'HAGAN (2004)
        
        This function's a-coefficients are chosen so that 5 of the input
        variables contribute significantly to the output variance, 5 have a 
        much smaller effect, and the remaining 5 have almost no effect on the 
        output variance. 
        
        Oakley, J. E., & O'Hagan, A. (2004). Probabilistic sensitivity analysis 
        of complex models: a Bayesian approach. Journal of the Royal Statistical
        Society: Series B (Statistical Methodology), 66(3), 751-769.
        
        y = oakley_ohagan_2004(x)
        
        Input:  
                x ... input data [N_input x 15] 
                      xi ~ N(μ=0, σ=1), for all i = 1, …, 15.
                
        Output: 
                y ... result [N_input x 1]
        """
    
        # load coefficients
        M = np.loadtxt('misc/oakley_ohagan_2004_M.txt')
        a1 = np.loadtxt('misc/oakley_ohagan_2004_a1.txt')
        a2 = np.loadtxt('misc/oakley_ohagan_2004_a2.txt')
        a3 = np.loadtxt('misc/oakley_ohagan_2004_a3.txt')

        # function
        y = np.array([(np.dot(self.parameters["x"],a1) + np.dot(np.sin(self.parameters["x"]),a2) + np.dot(np.cos(self.parameters["x"]),a3) \
            + np.sum(np.multiply(np.dot(self.parameters["x"],M),self.parameters["x"]),axis=1))]).transpose()

        return y
    
class Welch1992:

    def __init__(self, parameters, context):
        super(Welch1992, self).__init__(parameters, context)

    def simulate(self, process_id):
        """ 20-dimensional test function of WELCH (1992)

        For input variable screening purposes, it can be found that some input 
        variables of this function have a very high effect on the output, 
        compared to other input variables. As Welch et al. (1992) point out, 
        interactions and nonlinear effects make this function challenging. 
        
        Welch, W. J., Buck, R. J., Sacks, J., Wynn, H. P., Mitchell, T. J., & 
        Morris, M. D. (1992). Screening, predicting, and computer experiments. 
        Technometrics, 34(1), 15-25.
        
        y = welch_1992(x)
        
        Input:  
                x ... input data [N_input x 20] 
                      xi ~ U(-0.5, 0.5), for all i = 1, …, 20.
                
        Output: 
                y ... result [N_input x 1]
        """

        y = np.array([(5.0*self.parameters["x"][:,11]/(1+self.parameters["x"][:,0]) + 5*(self.parameters["x"][:,3]-self.parameters["x"][:,19])**2 + self.parameters["x"][:,4] + 40*self.parameters["x"][:,18]**3 \
            + 5*self.parameters["x"][:,18] + 0.05*self.parameters["x"][:,1] + 0.08*self.parameters["x"][:,2] - 0.03*self.parameters["x"][:,5] + 0.03*self.parameters["x"][:,6] \
            - 0.09*self.parameters["x"][:,8] - 0.01*self.parameters["x"][:,9] - 0.07*self.parameters["x"][:,10] + 0.25*self.parameters["x"][:,12]**2 - 0.04*self.parameters["x"][:,13] \
            + 0.06*self.parameters["x"][:,14] - 0.01*self.parameters["x"][:,16] - 0.03*self.parameters["x"][:,17])]).transpose()
    
        return y

class WingWeights(AbstractModel):

    def __init__(self, parameters, context):
            super(WingWeights, self).__init__(parameters, context)

        # copied from 'sphere.py' -> potential_3layers_surface_electrodes

    def simulate(self, process_id):
        """ 10-dimensional test function which models a light aircraft wing
        
        Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via 
        surrogate modelling: a practical guide. Wiley.
        
        y  = wing_weight(x)
        
        Input:  
                x ... input data [N_input x 10] 
                      x1(Sw)  ∈ [150, 200]
                      x2(Wfw) ∈ [220, 300]
                      x3(A)   ∈ [6, 10]
                      x4(Λ)   ∈ [-10, 10]
                      x5(q)   ∈ [16, 45]
                      x6(λ)   ∈ [0.5, 1]
                      x7(tc)  ∈ [0.08, 0.18]
                      x8(Nz)  ∈ [2.5, 6]
                      x9(Wdg) ∈ [1700, 2500]
                      x10(Wp) ∈ [0.025, 0.08]
                
        Output: 
                y ... result [N_input x 1]
        """
        y = np.array([( 0.036*self.parameters["x"][:,0]**0.758 * self.parameters["x"][:,1]**0.0035 \
          * (self.parameters["x"][:,2]/np.cos(self.parameters["x"][:,3])**2)**0.6 * self.parameters["x"][:,4]**0.006 * self.parameters["x"][:,5]**0.04 \
          * (100*self.parameters["x"][:,6]/np.cos(self.parameters["x"][:,3]))**-0.3 * (self.parameters["x"][:,7]*self.parameters["x"][:,8])**0.49 \
          + self.parameters["x"][:,0]*self.parameters["x"][:,9])]).transpose()
    
        return y

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