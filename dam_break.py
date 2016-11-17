# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 00:29:41 2016

@author: ajay
"""

import numpy as np
import matplotlib.pyplot as plt

def create_solid_particles(n,dx,dy,w,h):
    n1 = n/2 + n%2
    n2 = n/2
    
    x_min = min(-0.5*dx  - (n1-1)*dx,-dx - (n2-1)*dx)
#    print x_min
    y_min = min(-0.5*dy  - (n1-1)*dy,-dy - (n2-1)*dy)
#    print y_min
    
    
    if n%2 == 0:
        w_new = w + n2*dx
        
        x1,y1 = np.mgrid[x_min:w_new+1e-4:dx,y_min:h+1e-4:dy]
        x2,y2 = np.mgrid[x_min + 0.5*dx:w_new+1e-4:dx,y_min + 0.5*dx:h+1e-4:dy]
    else:
        w_new = w + n2*dx +0.5*dx
        
        x1,y1 = np.mgrid[x_min:w_new+1e-4:dx,y_min:h+1e-4:dy]
        x2,y2 = np.mgrid[x_min + 0.5*dx:w_new+1e-4:dx,y_min + 0.5*dx:h+1e-4:dy] 
        
        
    x1r = np.ones_like(x1)
    y1r = np.ones_like(y1)
    x2r = np.ones_like(x2)
    y2r = np.ones_like(y2)
        
    x1r[n1:-n1,n1:] = 0
    y1r[n1:-n1,n1:] = 0
    x2r[n2:-n2,n2:] = 0
    y2r[n2:-n2,n2:] = 0
    
    x1r = np.array(x1r,dtype = bool)
    y1r = np.array(y1r,dtype = bool)
    x2r = np.array(x2r,dtype = bool)
    y2r = np.array(y2r,dtype = bool)
    
    x1 = x1[x1r]
    y1 = y1[y1r]
    x2 = x2[x2r]
    y2 = y2[y2r]
    
#    x1 = x1.ravel()
#    y1 = y1.ravel()
#    x2 = x2.ravel()
#    y2 = y2.ravel()
    
    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))
    
    return x,y

def create_fluid_particles(dx,dy,w,h):
    x1,y1 = np.mgrid[0:w+1e-4:dx,0:h+1e-4:dy]
    x2,y2 = np.mgrid[0.5*dx:w+1e-4:dx,0.5*dy:h+1e-4:dy]
    x1 = x1.ravel()
    y1 = y1.ravel()
    x2 = x2.ravel()
    y2 = y2.ravel()
    
    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))
#    z = x+1j*y
    
    return x,y

def kernel(xi,xj,h,form):
    q = np.abs(xi-xj)/h
    Wij = form(q,h)
    return Wij
    
def spline_kernel(q,h):
    if q < 1.0:
        Wij = (2/(3*h))*(1 - (1.5*q**2)*(1 - q/2))
    elif 1.0 <= q < 2.0:
        Wij = (1/(6*h))*(2 - q)**3
    else:
        Wij = 0
        
    return Wij
    
def derivative_kernel(xi,xj,h,form):
    q = np.abs(xi-xj)/h
    dq = (1.0/h)*np.sign(xi - xj)
    DWij = form(q,h)*dq
    return DWij
    
def der_spline(q,h):
    if q < 1.0:
        DWij = (1.0/h)*(1.5*q**2 - 2.0*q)
    elif 1.0 <= q < 2.0:
        DWij = -(0.5/h)*(2.0 - q)**2
    else:
        DWij = 0
    return DWij

def main():
    w_solid = 4
    h_solid = 4
    w_fluid = 1
    h_fluid = 2

    dx = 0.012
    dy = 0.012
    h = 0.039
    
    dt = 1e-4  #remember to check this
    t = 3*dt
    
    x_fluid,y_fluid = create_fluid_particles(dx,dy,w_fluid,h_fluid)
    x_solid,y_solid = create_solid_particles(2,dx,dy,w_solid,h_solid)
    
    N_solid = len(x_solid)
    N_fluid = len(x_fluid)
    N = N_fluid + N_solid
    
    x = np.concatenate((x_fluid,x_solid))
    y = np.concatenate((y_fluid,y_solid))
    
    time_steps = dt/t +1
    
    rho = np.zeros((time_steps,N)) 
    p = np.zeros((time_steps,N))
    u = np.zeros((time_steps,N))
    u = np.zeros((time_steps,N))
    
    rho[0] = 1000*np.ones_like(x)
    p[0] = (1.013e5)*np.ones_like(x)
    
    
#main()
    
def test_create_fluid_particles():
    
    w_fluid = 1
    h_fluid = 2

    dx = 0.012
    dy = 0.012
    
    x,y = create_fluid_particles(dx,dy,w_fluid,h_fluid)
    
    plt.figure()
    plt.xlim(0,4)
    plt.ylim(0,4)
    plt.plot(x,y,'.')
    
test_create_fluid_particles()
    
    
def test_create_solid_particles():
    
    w_solid = 4
    h_solid = 4

    dx = 0.012
    dy = 0.012
    
    x,y = create_solid_particles(2,dx,dy,w_solid,h_solid)
    print len(x)
    print len(y)
    
#    plt.figure()
    plt.xlim(0,4.5)
    plt.ylim(0,4.5)
    plt.plot(x,y,'.')
    
test_create_solid_particles()
    
    