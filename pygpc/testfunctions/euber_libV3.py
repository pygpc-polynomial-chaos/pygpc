# Version 3 with steiner implenentation for non-mid section 2
import sympy as sp
from sympy import Matrix as syma
from sympy import diff
import numpy as np
import scipy.optimize
import pdb
import mpmath as mp
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


class euber():
 
    def __init__(self):
        eigenvalues_of_simple_bridge = [4.7300407449, 7.8532046241, 10.9956078382, 14.1371654913, 17.2787596574, 20.4203522456]
        # sympy stuff
        self.x = sp.symbols('x')
        tau1, tau2 = sp.symbols('tau1 tau2')
        tau11, tau12, tau13 = sp.symbols('tau11 tau12 tau13')
        tau21, tau22, tau23 = sp.symbols('tau21 tau22 tau23')
        # L_1, L_2, L_3 = sp.symbols('L_1 L_2  L_3')
        # E_1, E_2, E_3 = sp.symbols('E_1 E_2 E_3')
        # I_1, I_2, I_3 = sp.symbols('I_1 I_2 I_3')
        # A_1, A_2, A_3 = sp.symbols('A_1 A_2 A_3')
        # rho_1, rho_2, rho_3 = sp.symbols('rho_1 rho_2 rho_3')

        self.Lsym_1, self.Lsym_2, self.Lsym_3 = sp.symbols('Lsym_1 Lsym_2 Lsym_3')
        self.Esym_1, self.Esym_2, self.Esym_3 = sp.symbols('Esym_1 Esym_2 Esym_3')
        self.Isym_1, self.Isym_2, self.Isym_3 = sp.symbols('Isym_1 Isym_2 Isym_3')

        # trancendet Ansatz functions for E-B-Beam with 3 sections and axial tension
        TRAN_axt = ([sp.cos(self.x*tau1), sp.sin(self.x*tau1), sp.cosh(self.x*tau2), sp.sinh(self.x*tau2)])
        self.X1_axt = [i.subs([(tau1, tau11)
                               ,(tau2, tau21)]) for i in TRAN_axt]
        self.X2_axt = [i.subs([(tau1, tau12)
                               ,(tau2, tau22)]) for i in TRAN_axt]
        self.X3_axt = [i.subs([(tau1, tau13)
                               ,(tau2, tau23)]) for i in TRAN_axt]

       #mpmath
        self.tau1_sec1_axt_mp = lambda omega, k1, p1: mp.sqrt(2)*mp.sqrt((mp.sqrt(4*k1**4*omega**2 + p1**4) - p1**2)/k1**4)/2 #sin&cos = LOI2
        self.tau1_sec2_axt_mp = lambda lambda_1,k1,k2,p1,p2: ((1/2)*mp.sqrt(-2*p2**4 + 2*mp.sqrt(4*k1**4*k2**4*lambda_1**4 + 4*k2**4*lambda_1**2*p1**4 + p2**8))/k2**2)
        self.tau1_sec3_axt_mp = lambda lambda_1,k1,k3,p1,p3: ((1/2)*mp.sqrt(-2*p3**4 + 2*mp.sqrt(4*k1**4*k3**4*lambda_1**4 + 4*k3**4*lambda_1**2*p1**4 + p3**8))/k3**2)

        self.tau2_sec1_axt_mp = lambda omega, k1, p1: mp.sqrt(2)*mp.sqrt((p1**2 + mp.sqrt(4*k1**4*omega**2 + p1**4))/k1**4)/2 #sinh&cosh = LOI1
        self.tau2_sec2_axt_mp = lambda lambda_1,k1,k2,p1,p2: ((1/2)*mp.sqrt(2*p2**4 + 2*mp.sqrt(4*k1**4*k2**4*lambda_1**4 - 4*k2**4*lambda_1**2*p1**4 + p2**8))/k2**2)
        self.tau2_sec3_axt_mp = lambda lambda_1,k1,k3,p1,p3: ((1/2)*mp.sqrt(2*p3**4 + 2*mp.sqrt(4*k1**4*k3**4*lambda_1**4 - 4*k3**4*lambda_1**2*p1**4 + p3**8))/k3**2)
        self.tau2_23_alt = lambda lambda_1,k1,k2,p1,p2: ((1/2)*mp.sqrt(2*p2**4 - 2*mp.sqrt(4*k1**4*k2**4*lambda_1**4 - 4*k2**4*lambda_1**2*p1**4 + p2**8))/k2**2)


        # Calculating lambda initially to avoid multiple calculations
        self.BCMatrix_3sec_axt_lambda = self.get_BCMatrix_3sec_axt_lambda()

        # Precision with mpmath calculations (Determinant and taus)
        self.prec = 50

    def define_simple_beam(self, b1=0.03, b2=0.05, b3=0.03,
                                 h1=0.02, h2=0.01, h3=0.02,
                                 L1=0.3, L2=0.7, L3=1.0,
                                 E1=200.0e9, E2=70.0e9, E3=200.0e9,
                                 rho1=8700.0, rho2=2700.0, rho3=8700.0,
                                 N=100):
        self.b_1 = b1
        self.b_2 = b2
        self.b_3 = b3
        self.h_1 = h1
        self.h_2 = h2
        self.h_3 = h3
        self.L_1 = L1
        self.L_2 = L2
        self.L_3 = L3
        self.E_1 = E1
        self.E_2 = E2
        self.E_3 = E3
        self.rho_1 = rho1
        self.rho_2 = rho2
        self.rho_3 = rho3
        self.N = N
        self.A_1, self.I_1 = self.get_area_and_inertia_rectangle(self.b_1, self.h_1)
        self.A_2, self.I_2 = self.get_area_and_inertia_rectangle(self.b_2, self.h_2)
        self.A_3, self.I_3 = self.get_area_and_inertia_rectangle(self.b_3, self.h_3)
        self.k_1, self.p_1 = self.get_k_and_p_value(self.A_1, self.I_1, E1, rho1, N)
        self.k_2, self.p_2 = self.get_k_and_p_value(self.A_2, self.I_2, E2, rho2, N)
        self.k_3, self.p_3 = self.get_k_and_p_value(self.A_3, self.I_3, E3, rho3, N)
        
    def define_layered_beam(self, b1=0.03, #b2=0.05, b3=0.03,            #Sections
                                  L1=0.3, L2=0.7, L3=1.0,               #Sections
                                  dA=0.02, dB=0.01, dC=0.02,            #Layers
                                  EA=200.0e9, EB=70.0e9, EC=200.0e9,    #Layers
                                  rhoA=8700.0, rhoB=2700.0, rhoC=8700.0,#Layers
                                  N=100):                               #uniform
        
        #Layers
        self.b_1 = b1
        self.b_2 = b1 #!!
        self.b_3 = b1 #!!
        self.h_1 = dA + dB + dC
        self.h_2 = dA + dB
        self.h_3 = self.h_1
        self.L_1 = L1
        self.L_2 = L2
        self.L_3 = L3
        self.d_A = dA
        self.d_B = dB
        self.d_C = dC

        #Sections
        self.E_A = EA
        self.E_B = EB
        self.E_C = EC
        self.rho_A = rhoA
        self.rho_B = rhoB
        self.rho_C = rhoC

        self.init_parameter_lists()
        
        #self.get_section_E([EA, EB, EC], [dA, dB, dC])
        self.I_1 = np.sum(self.I_layers123_steiner)
        self.I_2 = np.sum(self.I_layers12_steiner)
        dist_of_neutral_layers = self.h_1/2 - self.h_2/2#- self.get_neutral_layer(self.ys[:2], self.A_layers[:2], self.E_layers[:2])
        #self.I_2 = self.I_2 + (self.d_C/2)**2 * np.sum(self.A_layers[:1])
        

        self.I_3 = self.I_1

        self.E_1 = np.sum(self.I_layers123_steiner*self.E_layers)/self.I_1
        #self.E_2 = np.sum(self.I_layers12_steiner*self.E_layers[:2])/np.sum(self.I_layers12_steiner)
        self.E_2 = np.sum(self.I_layers12_steiner*self.E_layers[:2])/self.I_2
        self.E_3 = self.E_1 # Seciton1 and 3 have same layers
        self.I_2 = self.I_2 + dist_of_neutral_layers**2 * np.sum(self.A_layers[:1])


        self.A_1 = np.sum(self.A_layers)
        self.A_2 = np.sum(self.A_layers[:2])
        self.A_3 = self.A_1 # Seciton1 and 3 have same layers
        self.rho_1 = np.sum(self.rho_layers_weighted)/self.A_1
        self.rho_2 = np.sum(self.rho_layers_weighted[:2])/self.A_2
        self.rho_3 = self.rho_1 # Seciton1 and 3 have same layers

        self.k_1, self.p_1 = self.get_k_and_p_value(self.A_1, self.I_1, self.E_1, self.rho_1, N)
        self.k_2, self.p_2 = self.get_k_and_p_value(self.A_2, self.I_2, self.E_2, self.rho_2, N)
        self.k_3, self.p_3 = self.get_k_and_p_value(self.A_3, self.I_3, self.E_3, self.rho_3, N)

    def init_parameter_lists(self):
        # Sections 123
        self.b = np.array([self.b_1, self.b_2, self.b_3])
        helper = self.d_A + self.d_B + self.d_C
        self.height = np.array([helper, helper - self.d_C, helper]) # only if A_1  = A3

        # Layers abc
        self.d = np.array([self.d_A, self.d_B, self.d_C])
        self.ys = self.get_centerpoints_of_layers_rectangle(*self.d) # center points of each Layer
        self.E_layers = np.array([self.E_A, self.E_B, self.E_C])
        self.A_layers, self.I_layers = self.get_area_and_inertia_rectangle(self.b, self.d)# only if b = [b1,b1,b1]=[bA,bB,bC]
        
        self.I_layers123_steiner = self.get_weighted_I_layer()
        self.I_layers12_steiner = self.get_weighted_I_layer(layers=[0, 1])

        self.rho_layers = np.array([self.rho_A, self.rho_B, self.rho_C])
        self.rho_layers_weighted = self.get_weighted_rho_layer()

    def get_weighted_I_layer(self, layers=[0, 1, 2]):
        neutral_layer = self.get_neutral_layer(self.ys[layers], self.A_layers[layers], self.E_layers[layers])
        offset_to_neutral_layer = self.ys[layers] - neutral_layer
        # print(neutral_layer)
        # print(offset_to_neutral_layer)
        return np.array(self.I_layers[layers] + offset_to_neutral_layer**2*self.A_layers[layers])
        
    def get_weighted_rho_layer(self):
        return self.rho_layers * self.A_layers

    # Systemmatrix for beam with 3 sections and axial tension
    # #######################################################
    def BCMatrix_3sec_axt(self):
        BC = sp.zeros(12,12)
        # Boundary Conditions left x==|--|==|
        BC[0,:] = syma([[i.subs(self.x,0) for i in self.X1_axt] + 
                        [0,0,0,0] + 
                        [0,0,0,0] ])  

        BC[1,:] = syma([[diff(i,self.x).subs(self.x,0) for i in self.X1_axt] +
                        [0,0,0,0] + 
                        [0,0,0,0] ])

        # Transition Conditions left |==x--|==|
        BC[2,:] = syma([[i.subs(self.x,self.Lsym_1) for i in self.X1_axt]  + 
                        [i.subs(self.x,self.Lsym_1)*(-1) for i in self.X2_axt] + 
                        [0,0,0,0] ])

        BC[3,:] = syma([[diff(i,self.x).subs(self.x,self.Lsym_1) for i in self.X1_axt]  + 
                        [diff(i,self.x).subs(self.x,self.Lsym_1)*(-1) for i in self.X2_axt] + 
                        [0,0,0,0] ])

        BC[4,:] = syma([[self.Esym_1*self.Isym_1*diff(i,self.x,self.x).subs(self.x,self.Lsym_1)*(-1)  for i in self.X1_axt] + 
                        [self.Esym_2*self.Isym_2*diff(i,self.x,self.x).subs(self.x,self.Lsym_1)  for i in self.X2_axt]+ 
                        [0,0,0,0] ])

        BC[5,:] = syma([[self.Esym_1*self.Isym_1*diff(i,self.x,self.x,self.x).subs(self.x,self.Lsym_1)  for i in self.X1_axt] + 
                        [self.Esym_2*self.Isym_2*diff(i,self.x,self.x,self.x).subs(self.x,self.Lsym_1)*(-1)  for i in self.X2_axt]+ 
                        [0,0,0,0] ])

        # Transition Conditions right |==|--x==|
        BC[6,:] = syma([[0,0,0,0] +
                        [i.subs(self.x,self.Lsym_2) for i in self.X2_axt] + 
                        [i.subs(self.x,self.Lsym_2)*(-1) for i in self.X3_axt]]) 

        BC[7,:] = syma([[0,0,0,0] +
                        [diff(i,self.x).subs(self.x,self.Lsym_2) for i in self.X2_axt] + 
                        [diff(i,self.x).subs(self.x,self.Lsym_2)*(-1) for i in self.X3_axt]]) 
        BC[8,:] = syma([[0,0,0,0] +
                        [self.Esym_2*self.Isym_2*diff(i,self.x,self.x).subs(self.x,self.Lsym_2)*(-1) for i in self.X2_axt] + 
                        [self.Esym_3*self.Isym_3*diff(i,self.x,self.x).subs(self.x,self.Lsym_2) for i in self.X3_axt]]) 
        BC[9,:] = syma([[0,0,0,0] +
                        [self.Esym_2*self.Isym_2*diff(i,self.x,self.x,self.x).subs(self.x,self.Lsym_2) for i in self.X2_axt] + 
                        [self.Esym_3*self.Isym_3*diff(i,self.x,self.x,self.x).subs(self.x,self.Lsym_2)*(-1) for i in self.X3_axt]]) 

        # Boundary Conditions right |==|--|==x
        BC[10,:] = syma([[0,0,0,0] +
                        [0,0,0,0] +  
                        [i.subs(self.x, self.Lsym_3) for i in self.X3_axt]]) 
        BC[11,:] = syma([[0,0,0,0] + 
                        [0,0,0,0] +
                        [diff(i,self.x).subs(self.x, self.Lsym_3) for i in self.X3_axt]]) 
        return BC

    def get_BCMatrix_3sec_axt_lambda(self):
        #smbls = ['tau11', 'tau21', 'tau12', 'tau22', 'tau13', 'tau23', 'E_1', 'E_2', 'E_3', 'I_1', 'I_2', 'I_3', 'L_1', 'L_2', 'L_3']
        smbls = ['tau11', 'tau21', 'tau12', 'tau22', 'tau13', 'tau23', 
                 'Esym_1', 'Esym_2', 'Esym_3', 
                 'Isym_1', 'Isym_2', 'Isym_3', 
                 'Lsym_1', 'Lsym_2', 'Lsym_3']
        symdet = self.BCMatrix_3sec_axt()
        return sp.utilities.lambdify(smbls, symdet, modules='mpmath')
    
    def calc_taus(self, omega):
        mp.mp.dps = self.prec
        
        # Section 1
        self.tau11 = mp.fabs(self.tau1_sec1_axt_mp(omega, self.k_1, self.p_1))
        self.tau21 = mp.fabs(self.tau2_sec1_axt_mp(omega, self.k_1, self.p_1))
        # Section 2
        self.tau12 = mp.fabs(self.tau1_sec2_axt_mp(self.tau11, self.k_1, self.k_2, self.p_1, self.p_2))
        self.tau22 = mp.fabs(self.tau2_sec2_axt_mp(self.tau21, self.k_1, self.k_2, self.p_1, self.p_2))
        # Section 3
        self.tau13 = mp.fabs(self.tau1_sec3_axt_mp(self.tau11, self.k_1, self.k_3, self.p_1, self.p_3))
        self.tau23 = mp.fabs(self.tau2_sec3_axt_mp(self.tau21, self.k_1, self.k_3, self.p_1, self.p_3))
    
    
    def get_BCMatrix_3sec_axt_det(self):
        mp.mp.dps = self.prec
        mat = self.BCMatrix_3sec_axt_lambda(self.tau11, self.tau21,
                                            self.tau12, self.tau22,
                                            self.tau13, self.tau23,
                                            self.E_1, self.E_2, self.E_3, 
                                            self.I_1, self.I_2, self.I_3, 
                                            self.L_1, self.L_2, self.L_3)
        det = mp.det(mat)
        det = np.float64( mp.log(mp.fabs(det))*mp.sign(det) )
        return det

    def peek_det(self, oms):
        mp.mp.dps = self.prec
        det = []
        for om in oms:
            self.calc_taus(om)
            det.append( self.get_BCMatrix_3sec_axt_det() )
        return np.array(det)

    def peek_det_scalar(self, om):
        mp.mp.dps = self.prec
        self.calc_taus(om)
        return self.get_BCMatrix_3sec_axt_det() 
        


    def lyrd_beam_reduced_params(self, b1, 
                                 l1, l2, l3, 
                                 dA, dB, dC, 
                                 EA=200.0e9, EB=70.0e9, EC=200.0e9,    #Layers
                                 rhoA=8700.0, rhoB=2700.0, rhoC=8700.0,#Layers
                                 N=1000):
        L1 = l1
        L2 = l1 + l2
        L3 = l1 + l2 + l3 
        self.define_layered_beam(b1,                                   #Sections
                                L1, L2, L3,                           #Sections
                                dA, dB, dC,                           #Layers
                                EA=EA, EB=EB, EC=EC,    #Layers
                                rhoA=rhoA, rhoB=rhoB, rhoC=rhoC,#Layers
                                N=N                               #uniform
                                )

    # search angular eigenfrequencies
    def get_first3_ekf(self):
        # does not work on random structures
        det = []
        om = 100.0
        while True:
            det.append(self.peek_det_scalar(om))
            if len(det)>1 and np.sign(det[-2]) != np.sign(det[-1]):
                om1 = scipy.optimize.brentq(self.peek_det_scalar,om-100,om)
                break
            om += 100
        om2 = scipy.optimize.brentq(self.peek_det_scalar, om1*2, om1*3, xtol=1e-2)
        om3 = scipy.optimize.brentq(self.peek_det_scalar, om1*3, om2*3, xtol=1e-2)
        return np.array([om1, om2, om3])

    def get_first_n_ekf_bruteforce(self, n=3, om=100.0, stepsize=100.0, brentq_xtol=1e-2):
        det = []
        oms = []        
        ekfs = []
        while True:
            det.append(self.peek_det_scalar(om))
            oms.append(om)
            if len(det)>1 and np.sign(det[-2]) != np.sign(det[-1]):
                ekfs.append( scipy.optimize.brentq(self.peek_det_scalar,oms[-2],oms[-1], xtol=brentq_xtol) )
                stepsize = ekfs[0]
                if len(ekfs) == n:
                    return np.array(ekfs), det, oms
                    
            om += stepsize

            
                    
            om += stepsize
    ### alternative eigenvalue calc
    def calc_taus_alt(self, omega):
        mp.mp.dps = self.prec
        
        # Section 1
        self.tau11 = mp.fabs(self.tau1_sec1_axt_mp(omega, self.k_1, self.p_1))
        self.tau21 = mp.fabs(self.tau2_sec1_axt_mp(omega, self.k_1, self.p_1))
        # Section 2
        self.tau12 = mp.fabs(self.tau1_sec2_axt_mp(self.tau11, self.k_1, self.k_2, self.p_1, self.p_2))
        self.tau22 = mp.fabs(self.tau2_sec2_axt_mp(self.tau21, self.k_1, self.k_2, self.p_1, self.p_2))
        # Section 3
        self.tau13 = mp.fabs(self.tau1_sec3_axt_mp(self.tau11, self.k_1, self.k_3, self.p_1, self.p_3))
        self.tau23 = mp.fabs(self.tau2_23_alt(self.tau21, self.k_1, self.k_3, self.p_1, self.p_3))

    def get_first_n_ekf_bruteforce_alt(self, n=3, om=100.0, stepsize=100.0, brentq_xtol=1e-2):
        det = []
        oms = []        
        ekfs = []
        while True:
            det.append(self.peek_det_scalar_alt(om))
            oms.append(om)
            if len(det)>1 and np.sign(det[-2]) != np.sign(det[-1]):
                ekfs.append( scipy.optimize.brentq(self.peek_det_scalar_alt,oms[-2],oms[-1], xtol=brentq_xtol) )
                stepsize = ekfs[0]
                if len(ekfs) == n:
                    return np.array(ekfs), det, oms
                    
            om += stepsize
    def peek_det_scalar_alt(self, om):
        mp.mp.dps = self.prec
        self.calc_taus_alt(om)
        return self.get_BCMatrix_3sec_axt_det() 
 
    

    ### Static Methods
    @staticmethod
    def get_approx_ekf(fist_3_ekfs, order, poly_deg=2):
        coef = np.polyfit(range(1, poly_deg+2), fist_3_ekfs, poly_deg)
        return np.polyval(coef, order)

    @staticmethod
    def get_area_and_inertia_rectangle(b, h):
        A = b * h
        Iz = b * h**3 / 12
        return A, Iz
    
    @staticmethod
    def get_k_and_p_value(A, Iz, E, rho, N):
        k = (E * Iz / rho / A )**0.25
        p = (N / rho / A)**0.5
        return k, p
       
    @staticmethod
    def get_centerpoints_of_layers_rectangle(*args):
        return np.array([d/2+sum(args[:i]) for i, d in enumerate(args)])

    @staticmethod
    def get_neutral_layer(ys, A_layers, E_layers):
        helper = np.array(A_layers * E_layers)
        return np.sum(ys * helper) / np.sum(helper)


def plot_beam_geometry(geom_param, hight_factor=10 , colors=['red', 'green', 'orange']):
    # Plotting Geometry
    b1, l1, l2, l3, hA, hB, hC = geom_param
    hA, hB, hC = hA*hight_factor, hB*hight_factor, hC*hight_factor
    rects = []
    # layerA
    rects.append(Rectangle((0, 0), l1+l2+l3, hA, color=colors[0], edgecolor=None,linewidth=0))
    # layerB
    rects.append(Rectangle((0, hA), l1+l2+l3, hB, color=colors[1], edgecolor=None, linewidth=0))
    # layerC
    rects.append(Rectangle((0, hA+hB), l1, hC, color=colors[2], edgecolor=None, linewidth=0))
    rects.append(Rectangle((l1+l2,hA+hB), l3, hC, color=colors[2], edgecolor=None, linewidth=0))

    fig, ax = plt.subplots(1, figsize=cm2inch((l1+l2+l3)*3.2, (hA+hB+hC)*1.2), subplot_kw={'xlim':(0,l1+l2+l3), 'ylim':(0,hA+hB+hC)})
    [ax.add_patch(p) for p in rects]
    ax.axis('off')

    return fig

    
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
# import time
# cycls = 200

# # init_time = time.time()
# # for _ in range(cycls):
# #     bb = euber()
# # init_time = (time.time() - init_time)/cycls
# bb = euber()
# definetime = time.time()
# for _ in range(cycls):
#     bb.define_layered_beam()
# definetime = (time.time() - definetime)/cycls

# dettime = time.time()
# for _ in range(cycls):
#     bb.calc_taus(30)
#     bb.get_BCMatrix_3sec_axt_det()
# dettime = (time.time() - dettime)/cycls

# print(definetime, dettime)

# beam = euber()
# beam.define_layered_beam()
# bendingstiffness = lambda E,I,y,A: np.sum( np.array(E)*(np.array(I)+np.array(y)**2*np.array(A)) )
# reduced_mass = lambda rho, A: np.sum(rho*A)

# E = beam.E_layers
# A = beam.A_layers
# layers = [0,1,2]
# helper = np.array(beam.A_layers[layers]) * np.array(beam.E_layers[layers])
# neutral_layer = np.sum(np.array(beam.ys[layers]) * helper) / np.sum(helper)
# offset_to_neutral_layer = beam.ys[layers] - neutral_layer
# I_layers = beam.b * beam.d**3/12
# EI_from_euber = beam.E_1*beam.I_1

# mss = reduced_mass(beam.rho_layers, A)
# print(f'oop={beam.rho_1 * beam.A_1}')
# print(f'bnd={mss}')
# 1+1



























