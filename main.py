# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:21:55 2023

@author: Lorraine
"""

import numpy as np
import matplotlib.pyplot as plt

# Gauss-Lobatto Legendre interpolants points
from GetGLL import GetGLL
from lagrange1st import lagrange1st 
from source import source

class SimulationSetup:
    def __init__(self, nt, L, N, NEL, Tdom, iplot, vs ,rho, el_span, percent):
        # Setting the different variables
        self.nt = nt
        self.L = L
        self.N = N
        self.NEL = NEL
        self.Tdom = Tdom
        self.iplot = iplot
        self.vs = vs
        self.rho = rho
        self.el_span = el_span
        self.percent = percent
        
class WaveEquationSolver1D:
    def __init__(self, setup):
        # Initialization of the GLL poinrs integration weight
        self.setup = setup
        # xi: N+1 coordinates [-1 1] of GLL points
        # w: integration weights at GLL locations
        self.xi, self.w = GetGLL(setup.N)
        # Length of elements
        self.le = self.L / setup.NEL        # Initializing the elementary matrices
        self.mass_matrix_elem = np.zeros(self.N+1, dtype =  float)
        self.stif_matrix_elem = np.zeros((self.N+1, self.N+1), dtype =  float)
        
    def initialize_gll_points(self):
        # Initializating the GLL points
        k = 0
        self.xg = np.zeros((self.N * self.NEL) + 1)
        self.xg[k] = 0
        
        for i in range(1, self.NEL + 1):
            for j in range(0, self.N):
                k = k + 1
                self.xg[k] = (i - 1) * self.le + 0.5 * (self.xi[j + 1] + 1) * self.le
        
        self.xg = self.xg
        #pass
    
    def compute_time_step(self):
        # Computing time step
        dxmin = min(np.diff(self.xg))  
        eps = 0.1           # Courant value
        self.dt = eps * dxmin / self.vs   # Global time step
        
        # Mapping 
        jacobian = self.le / 2 
        jacobian_inv = 1 / jacobian    # Inverse Jacobian
    
        # 1st derivative of Lagrange polynomials
        l1d = lagrange1st(self.N)   # Array with GLL as columns for each N+1 polynomial
        #pass
        return jacobian, jacobian_inv, l1d
    
    def initialize_elastic_parameters(self):
        # ... Initialize elastic parameters
        el_span = 20                     # Number of elements spanning the Low velocity zone
        percent = 0.3                    # percentage of velocity reduction 
        a = el_span * self.N + 1         # width of the fault zone in grid points
        b = 375
        vs  = self.vs * np.ones((self.N * self.NEL +1))
        rho = self.rho * np.ones((self.N * self.NEL +1))
        
        # Applying of S-wave velocity reduction
        vs[b-int(a/2):b+int(a/2)] = max(vs) * percent
        mu  = rho * vs**2                # Shear modulus mu
        #pass
        return rho, vs, mu
    
    def global_mass_matrix(self, rho, vs, jacobian):
        # Assembling the mass matrix
        k = -1
        m = -1
        self.NG = (self.NEL - 1) * self.N + self.N + 1
        self.mass_matrix_glob = np.zeros(2 * self.NG) 
        
        for i in range(1, self.NEL+1): 
            # ------------------------------------
            # Elemental Mass matrix
            # ------------------------------------
            for l in range(0, self.N+1):
                m += 1
                self.mass_matrix_elem[l] = rho[m] * self.w[l] * self.jacobian    #stored as a vector since it's diagonal
            m -= 1 
            # ------------------------------------
            for j in range(0, self.N+1): 
                k = k + 1
                if i>1:
                    if j==0:
                        k = k - 1
                self.mass_matrix_glob[k] = self.mass_matrix_glob[k] + self.mass_matrix_elem[j]
        
        # Inverse matrix of M 
        # --------------------------------------------------------------- 
        self.mass_matrix_inv = np.identity(self.NG)
        for i in range(0, self.NG):
            self.mass_matrix_inv[i,i] = 1./ self.mass_matrix_glob[i]
        pass
    
    def global_stiffness_matrix(self, mu, jacobian, jacobian_inv, l1d):
        # Assembling the stiffness matrix
        
        self.stif_matrix_glob = np.zeros([self.NG, self.NG])
        xe = 0 
        
        for e in range(1, self.NEL + 1):
            i0 = (e - 1) * self.N + 1
            j0 = i0
            # ------------------------------------
            # Elemental Stiffness Matrix
            # ------------------------------------
            for i in range(-1,self. N):
                for j in range(-1, self.N):
                    sum = 0
                    for k in range(-1, self.N):                
                        sum = sum + mu[k+1+xe] *self. w[k+1] * jacobian_inv**2 * jacobian * self.l1d[i+1,k+1] * self.l1d[j+1,k+1]
                    self.stif_matrix_elem[i+1, j+1] = sum    
            xe += self.N
        
            for i in range(-1, self.N):
                for j in range(-1, self.N):
                    self.stif_matrix_glob[i0+i, j0+j] += self.stif_matrix_elem[i+1, j+1]

        pass
    
    def solve_wave_equation(self, rho, vs):
        # Solving the wave equation
        src = source(self.dt, self.Tdom)
        isrc = 200
        NG = self.NG
        
        # Initialization of the displacement
        xg = self.xg
        UG = np.zeros(NG)
        UG_old = UG
        UG_new = UG
        F = UG
        
        # BC: absorbing
        tabs = np.zeros(NG)
        isrc = 200  # Source location
        
        x_t = []
        
        for it in range(self.setup.nt):
            # Calculate source function, boundary conditions, and update UG_new
            # Source initialization
            F = np.zeros(NG)
            if it < len(src):
                F[isrc-1] = src[it-1]
                
            # Absorbing boundary condition at z = 0
            tabs[0] =  rho[0] * vs[0] * (UG[0] - UG_old[0])/(self.dt)  
            
            # Absorbing boundary condition at z = end
            tabs[NG-1] =  rho[NG-1] * vs[NG-1] * (UG[NG-1] - UG_old[NG-1])/(self.dt)
                
            
            # Time extrapolation
            UG_new = self.dt**2 * self.mass_matrix_inv @ (F - self.stif_matrix_glob @ UG-tabs) + 2 * UG - UG_old
            UG_old, UG = UG, UG_new
            x_t.append(UG) 

        x_t = np.array(x_t)
        
        return xg, UG, UG_old, UG_new
        #pass

class WaveAnimation:
    def __init__(self, xg, UG, UG_old, UG_new, iplot, nt, dt):
        self.xg = xg
        self.UG = UG
        self.UG_old = UG_old
        self.UG_new = UG_new
        self.iplot = iplot
        self.nt = nt
        self.dt = dt
        
        self.fig = None
        self.ax = None
        self.lines = None
    
    def initialize_animation_plot(self):
        # Initialization of the animation plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(0, self.xg[-1])
        self.ax.set_ylim(min(self.UG), max(self.UG))
        self.ax.set_title('Wave propagation 1D Spectral Element Method')
        self.ax.set_xlabel('Depth (m)')
        self.ax.set_ylabel('Displacement u(z,t)')
        self.lines = self.ax.plot(self.xg, self.UG, color="black", lw=1)
        
        pass
    
    def update_animation(self, it, lines):
        # ... Update animation
        self.lines.set_ydata(self.UG)
        self.ax.set_title(f'Wave propagation 1D Spectral Element Method (Time step: {it})')
        self.fig.canvas.draw()
        plt.pause(0.001)
        
        pass
    
    def animate(self):
        # Animation of the plot
        self.initialize_animation_plot()

        for it in range(self.nt):
            # Update UG, UG_old, UG_new using the solver's logic
            # Call the solver's update method
            
            if not it % self.iplot:
                self.update_animation(it)
        pass


def main():
    setup = SimulationSetup(
        nt=7500, L=8000., N=4, NEL=150, Tdom=0.4, iplot=30,
        vs=2500., rho=2000, el_span=20, percent=0.3
    )
    
    solver = WaveEquationSolver1D(setup)
    solver.initialize_gll_points()
    solver.compute_time_step()
    solver.initialize_elastic_parameters()
    solver.global_mass_matrix()
    solver.global_stiffness_matrix()
    xg, UG, UG_old, UG_new = solver.solve_wave_equation()
    
    animation = WaveAnimation(xg, UG, UG_old, UG_new, setup.iplot, setup.nt, solver.dt)
    animation.initialize_animation_plot()
    animation.animate()

if __name__ == "__main__":
    main()