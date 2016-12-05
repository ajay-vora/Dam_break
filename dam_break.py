# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 00:29:41 2016

@author: ajay
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

def kernel(xi,xj,yi,yj,h,form):
    q = np.sqrt((xi-xj)**2 + (yi-yj)**2)/h
    Wij = form(q,h)
    return Wij
    
def spline_kernel(q,h):
    sigma = 10.0/(7.0*np.pi*h**2)
    if q < 1.0:
        Wij = sigma*(1.0 - (1.5*q**2)*(1.0 - q/2.0))
    elif 1.0 <= q < 2.0:
        Wij = sigma*0.25*(2.0 - q)**3
    else:
        Wij = 0
        
    return Wij
    
def gauss_kernel(q,h):
    sigma = (1.0/(np.pi*h**2))
    if q < 3.0:
        Wij = sigma*np.exp(-q**2)
    else: 
        Wij = 0.0
    return Wij
    
def derivative_kernel(xi,xj,yi,yj,h,form):
    r = np.sqrt((xi-xj)**2 + (yi-yj)**2)
    
    if r != 0.0:
        q = r/h
        dq_x = (xi - xj)/r
        dq_y = (yi - yj)/r
        DW_q = form(q,h)  
        DWx_ij = DW_q*dq_x
        DWy_ij = DW_q*dq_y
        
    else:
        DWx_ij = 0.0
        DWy_ij = 0.0
        
    return DWx_ij,DWy_ij
    
def der_spline(q,h):
    sigma = 10.0/(7.0*np.pi*h**2)
    if q < 1.0:
        DW_q = sigma*((9.0/4.0)*q**2 - 3.0*q)
    elif 1.0 <= q < 2.0:
        DW_q = -sigma*(3.0/4.0)*(2.0 - q)**2
    else:
        DW_q = 0
    return DW_q
    
def der_gauss(q,h):
    sigma = (1.0/(np.pi*h**2))
    if q < 3.0:
        DW_q = sigma*(-2*q)*np.exp(-q**2)
    else: 
        DW_q = 0.0
    return DW_q

def create_all_particles(w_solid = 4, h_solid = 4, w_fluid = 1, h_fluid = 2, \
dx = 0.012, dy = 0.012, n_solid_layers = 2):
    
    x_fluid,y_fluid = create_fluid_particles(dx,dy,w_fluid,h_fluid)
    x_solid,y_solid = create_solid_particles(n_solid_layers,dx,dy,w_solid,\
    h_solid)

    N_solid = len(x_solid)
    N_fluid = len(x_fluid)
    
    x = np.concatenate((x_solid,x_fluid))
    y = np.concatenate((y_solid,y_fluid))
    
    return x,y,N_solid,N_fluid
    
def sph_equations(m, rho, p, u, v, x, y, h, N, N_solid):
    
    rhs_rho = np.zeros(N)
#    rhs_p = np.zeros(N)
    rhs_u = np.zeros(N)
    rhs_v = np.zeros(N)
    xsph = np.zeros(N)
    ysph = np.zeros(N)
    
    for i in range(N):
#        v_term = np.zeros(N)
        for j in range(N):
            
            Wij = kernel(x[i],x[j],y[i],y[j],h,gauss_kernel)                
            DWx_ij, DWy_ij = derivative_kernel(x[i],x[j],y[i],y[j],h,\
            der_gauss)
            
            rhs_rho[i] += rho[i]*(m[j]/rho[j])*((u[i]-u[j])*DWx_ij + \
            (v[i]-v[j])*DWy_ij)
            
            if i not in range(N_solid): 
                v_term = -m[j]*((p[i]/rho[i]**2) + (p[j]/rho[j]**2))
#                print v_term
                rhs_u[i] += v_term*DWx_ij
                rhs_v[i] += v_term*DWy_ij + (-9.8)
                
                rho_ij = 0.5*(rho[i] + rho[j])
                xsph = -0.5*m[j]*((u[i]-u[j])/rho_ij)*Wij
                ysph = -0.5*m[j]*((v[i]-v[j])/rho_ij)*Wij 
            
    return rhs_rho,rhs_u,rhs_v,xsph,ysph
    

def main():
    
    h = 0.39
    
    dt = 0.004  #Courant number = 0.3 , u = 6.26 yields dt = 0.00575
    t = 3*dt
    time_steps = int(np.ceil(t/dt + 1))
    dx = 0.12
    dy = 0.12
     
    x1,y1,N_solid,N_fluid = create_all_particles(dx = 0.12,dy = 0.12) 

    N = N_solid + N_fluid
    
    rho = np.zeros((time_steps,N))
    p = np.zeros((time_steps,N))
    u = np.zeros((time_steps,N))
    v = np.zeros((time_steps,N))
    m = np.zeros(N)
    x = np.zeros((time_steps,N))
    y = np.zeros((time_steps,N))
    
#    m[:] = rho[0,0]*dx*dy
    rho[0] = 1000.0*np.ones(N)
    m[:] = rho[0,0]*dx*dy
    p[0] = (1.013e5)*np.ones(N)
    x[0] = x1
    y[0] = y1
    B = 1.013e5
    gamma = 7.0
    
    for i in range(1,time_steps):
        
#        p = B*((rho[i]/rho[0])**gamma - 1)
        
        rhs_rho, rhs_u, rhs_v, xsph, ysph = sph_equations(m, rho[i-1], p[i-1],\
        u[i-1], v[i-1], x[i-1], y[i-1], h, N, N_solid)
        
        rho[i] = rho[i-1] + dt*rhs_rho
        u[i] = u[i-1] + dt*rhs_u
        v[i] = v[i-1] + dt*rhs_v
        x[i] = x[i-1] + dt*(u[i-1] + xsph)
        y[i] = y[i-1] + dt*(v[i-1] + ysph)
        p[i] = B*((rho[i]/rho[0])**gamma - 1) #+ B
        
    return rho,p,u,v,x,y,N_solid

def plot_it(i=-1):
    
    rho,p,u,v,x,y,N_solid = main()
    
    plt.figure()
    plt.plot(x[i,0:N_solid],y[i,0:N_solid],'g.')
    plt.plot(x[i,N_solid:],y[i,N_solid:],'b.')
    
#plot_it()
    
def test_create_fluid_particles():
    
    w_fluid = 1
    h_fluid = 2

    dx = 0.012
    dy = 0.012
    
    x,y = create_fluid_particles(dx,dy,w_fluid,h_fluid)
    
    plt.figure()
    plt.xlim(0,4.5)
    plt.ylim(0,4.5)
    plt.plot(x,y,'.')
    
#test_create_fluid_particles()
    
    
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
    
#test_create_solid_particles()

def test_create_all_particles():
    
    x,y,N_solid,N_fluid = create_all_particles()
    
    plt.figure()
    plt.plot(x[0:N_solid],y[0:N_solid],'g.')
    plt.plot(x[N_solid:],y[N_solid:],'b.')
    
#test_create_all_particles()
    
def test_kernel():
    X,Y = np.mgrid[-4:4.0 + 1e-7:400j,-4:4.0 + 1e-7:400j]
    n,m = X.shape
    x = X.ravel()
    y = Y.ravel()
    
    h = 2
    z = np.zeros_like(x)

    for i in range(len(x)):
        
        z[i] = kernel(x[i],0.0,y[i],0.0,h,spline_kernel)


    x.shape = (n,m)
    y.shape = (n,m)
    z.shape = (n,m)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot_surface(x, y, z, rstride=10, cstride=10,linewidth=0, \
    antialiased=False)
    
#    ax.set_zlim(-0.01, 0.13)
    
#test_kernel()
    
    
def test_der_kernel():
    x = [0.4,-0.4,0.0,0.0]
    y = [0.0,0.0,0.4,-0.4]
    
    h = 2
    
    DWx1,DWy1 = derivative_kernel(x[0],0.0,y[0],0.0,h,der_spline)
    assert DWx1 < 0.0
    
    DWx2,DWy2 = derivative_kernel(x[1],0.0,y[1],0.0,h,der_spline)
    assert DWx2 > 0.0
    
    DWx3,DWy3 = derivative_kernel(x[2],0.0,y[2],0.0,h,der_spline)
    assert DWy3 < 0.0
    
    DWx4,DWy4 = derivative_kernel(x[3],0.0,y[3],0.0,h,der_spline)
    assert DWy4 > 0.0
    

#test_der_kernel()

    
