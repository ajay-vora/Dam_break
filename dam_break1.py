# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:54:36 2016

@author: ajay
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def create_stg_solid_particles(n,dx,dy,w,h):
    n1 = n/2 + n%2
    n2 = n/2
    
    if n%2 == 0.0:
        x_min = -n2*dx
        y_min = -n2*dy
        w_new = w + n2*dx
        
        x1,y1 = np.mgrid[x_min:w_new+1e-10:dx,y_min:h+1e-10:dy]
        x2,y2 = np.mgrid[x_min + 0.5*dx:w_new+1e-10:dx,y_min + 0.5*dx:h+1e-10:
                         dy] 
    else:
        
        x_min = -0.5*dx  - (n1-1)*dx
        y_min = -0.5*dy  - (n1-1)*dy
        w_new = w + n2*dx +0.5*dx
        
        x2,y2 = np.mgrid[x_min:w_new+1e-10:dx,y_min:h+1e-10:dy]
        x1,y1 = np.mgrid[x_min + 0.5*dx:w_new+1e-10:dx,y_min + 0.5*dx:h+1e-10:
                         dy]
        
    x1r = np.ones_like(x1)
    y1r = np.ones_like(y1)
    x2r = np.ones_like(x2)
    y2r = np.ones_like(y2)
    
    index1 = np.where(x1[:,0]>4.0)
    index2 = np.where(x2[:,0]>4.0)
    n3 = len(index1[0])
    n4 = len(index2[0])

    x1r[n2:-n3,n2:] = 0
    y1r[n2:-n3,n2:] = 0
    x2r[n1:-n4,n1:] = 0
    y2r[n1:-n4,n1:] = 0

    x1r = np.array(x1r,dtype = bool)
    y1r = np.array(y1r,dtype = bool)
    x2r = np.array(x2r,dtype = bool)
    y2r = np.array(y2r,dtype = bool)

    x1 = x1[x1r]
    y1 = y1[y1r]
    x2 = x2[x2r]
    y2 = y2[y2r]

    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))

    return x,y

def create_non_stg_solid_particles(n,dx,dy,w,h):

    xmin = -n*dx
    ymin = -n*dy

    w_new = w - xmin + 1e-10
    h_new = h - ymin + 1e-10

    x,y = np.mgrid[xmin:w_new:dx,ymin:h_new:dy]

    xr = np.ones_like(x)
    yr = np.ones_like(x)

    xr[n:-n,n:] = 0
    yr[n:-n,n:] = 0

    xr = np.array(xr,dtype = bool)
    yr = np.array(yr,dtype = bool)

    x = x[xr]
    y = y[yr]

    return x,y

def create_stg_fluid_particles(dx,dy,w,h):
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


def create_non_stg_fluid_particles(dx,dy,w,h):
    x,y = np.mgrid[0:w+1e-4:dx,0:h+1e-4:dy]
    x = x.ravel()
    y = y.ravel()

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


def create_all_particles(n_solid_layers = 3, dx = 0.012, dy = 0.012,
                         w_solid = 4.0, h_solid = 4.0,w_fluid = 1.0,
                         h_fluid = 2.0):

    x_fluid,y_fluid = create_non_stg_fluid_particles(dx,dy,w_fluid,h_fluid)
    x_solid,y_solid = create_stg_solid_particles(n_solid_layers,dx,dy,
                                                     w_solid,h_solid)
                                                     
    x_solid[:] = x_solid[:] - 0.5*dx #to be used with non_stg_fluid, solid_stg
    y_solid[:] = y_solid[:] - 0.5*dy #to be used with non_stg_fluid, solid_stg
    
    N_solid = len(x_solid)
    N_fluid = len(x_fluid)

    x = np.concatenate((x_solid,x_fluid))
    y = np.concatenate((y_solid,y_fluid))

    return x,y,N_solid,N_fluid


def hg_correction(rho):
    index = np.where(rho < 1000.0)
    rho[index] = 1000.0

    return rho


def sph_equations(m, rho, p, u, v, x, y, h, N, N_solid, r0, D):

    rhs_rho = np.zeros(N)
#    density = np.zeros(N)
#    rhs_p = np.zeros(N)
    rhs_u = np.zeros(N)
    rhs_v = np.zeros(N)
    xsph = np.zeros(N)
    ysph = np.zeros(N)
    c0 = 62.61 #10*u_max
    gamma1 = 0.5*(7.0 -1.0)
    alpha = 1
    beta = 0

    for i in range(N):
#        v_term = 0
        for j in range(N):

            Wij = kernel(x[i],x[j],y[i],y[j],h,spline_kernel)
            DWx_ij, DWy_ij = derivative_kernel(x[i],x[j],y[i],y[j],h,
                                               der_spline)

            u_ij = u[i]-u[j]
            v_ij = v[i]-v[j]

            rhs_rho[i] += rho[i]*(m[j]/rho[j])*(u_ij*DWx_ij + v_ij*DWy_ij)
#            density[i] += m[j]*Wij

            if i not in range(N_solid):
                v_term = -m[j]*((p[i]/rho[i]**2) + (p[j]/rho[j]**2))
#                print v_term

#                if j in range(N_solid):
#                    x_ij = x[i] - x[j]
#                    y_ij = y[i] - y[j]
#                    r_ij = np.sqrt(x_ij**2 + y_ij**2)
#
#                    if r0/r_ij >= 1.0:
#                        f = D*((r0/r_ij)**12 - (r0/r_ij)**4)
#                        f_x = f*(x_ij/r_ij**2)
#                        f_y = f*(y_ij/r_ij**2)
##                        print 'f_x = {} & f_y = {}'.format(f_x,f_y)
#                    else:
#                        f_x = 0.0
#                        f_y = 0.0
#
#                else:
#                    f_x = 0.0
#                    f_y = 0.0

                xij = x[i] - x[j]
                yij = y[i] - y[j]
                v_ijdotr_ij = u_ij*xij + v_ij*yij
                rho_ij = 0.5*(rho[i] + rho[j])

                if v_ijdotr_ij <= 0.0:

#               ADD ci, cj and cij
                    ci = c0*(rho[i]/1000.0)**gamma1
                    cj = c0*(rho[j]/1000.0)**gamma1
                    cij = 0.5*(ci + cj)

                    mu_ij = (h*v_ijdotr_ij)/(xij**2 + yij**2 + (0.1*h)**2)
                    pi_ij = (-alpha*cij*mu_ij + beta*mu_ij**2)/rho_ij
                else:
                    pi_ij = 0.0
#

                rhs_u[i] += (v_term - m[j]*pi_ij)*DWx_ij #+ f_x
#                print f_x
                rhs_v[i] += (v_term - m[j]*pi_ij)*DWy_ij  #+ f_y
#                print f_y

#                rho_ij = 0.5*(rho[i] + rho[j])
                xsph[i] += -0.5*m[j]*(u_ij/rho_ij)*Wij
                ysph[i] += -0.5*m[j]*(v_ij/rho_ij)*Wij

    return rhs_rho,rhs_u,rhs_v,xsph,ysph
#    return density,rhs_u,rhs_v,xsph,ysph


def main(n_iter=100):

    h = 0.39

    dt = 0.004  #Courant number = 0.3 , u_max = 6.26 yields dt = 0.00575
    t = n_iter*dt
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
    p[0] = np.ones(N)  #(1.013e5)*
    x[0] = x1
    y[0] = y1
    B = 560000.0  #1.013e5 # B = (rho_0*c_o**2) /gamma and
#                            c_0 = 10*max(u_max,np.sqrt(gL)) where is L is
#                            characterisitic vertical dimension of flow
    gamma = 7.0

    f_y = np.zeros(N)
    f_y[N_solid:] = -9.8*np.ones(N_fluid)

    r0 = dx
    D = 36.0

    for i in range(1,time_steps):

#        p = B*((rho[i]/rho[0])**gamma - 1)

        rhs_rho, rhs_u, rhs_v, xsph, ysph = sph_equations(m, rho[i-1], p[i-1],
                                                          u[i-1], v[i-1],
                                                          x[i-1], y[i-1], h, N,
                                                          N_solid, r0, D)

#        rho[i], rhs_u, rhs_v, xsph, ysph = sph_equations(m, rho[i-1], p[i-1],
#                                                         u[i-1], v[i-1],
#                                                         x[i-1], y[i-1], h, N,
#                                                         N_solid, r0, D)

        rho[i] = rho[i-1] + dt*rhs_rho
        u[i] = u[i-1] + dt*rhs_u
        v[i] = v[i-1] + dt*(rhs_v + f_y)
        x[i] = x[i-1] + dt*(u[i-1] + xsph)
        y[i] = y[i-1] + dt*(v[i-1] + ysph)

        rho[i] = hg_correction(rho[i])
        p[i] = B*((rho[i]/rho[0])**gamma - 1.0) #+ B

    return rho,p,u,v,x,y,N_solid


def plot_res(sol, i=-1):
    rho,p,u,v,x,y,N_solid = sol
    plt.figure()
    plt.plot(x[i,0:N_solid],y[i,0:N_solid],'g.')
    plt.plot(x[i,N_solid:],y[i,N_solid:],'b.')
    plt.show()


def plot_it(i=-1):

    plot_res(main(), -1)

#plot_it()


def test_create_fluid_particles(dx = 0.12,dy = 0.12,w_fluid = 1.0,
                                h_fluid = 2.0):

    x,y = create_stg_fluid_particles(dx,dy,w_fluid,h_fluid)
    x1,y1 = create_non_stg_fluid_particles(dx,dy,w_fluid,h_fluid)

    plt.figure()
    plt.xlim(-0.5,4.5)
    plt.ylim(-0.5,4.5)
    plt.plot(x,y,'.')

    plt.figure()
    plt.xlim(-0.5,4.5)
    plt.ylim(-0.5,4.5)
    plt.plot(x1,y1,'.')
#test_create_fluid_particles()



def test_create_solid_particles(n=3, dx = 0.12, dy = 0.12, w_solid = 4.0,
                                h_solid = 4.0):

    x,y = create_stg_solid_particles(n,dx,dy,w_solid,h_solid)
    x1,y1 = create_non_stg_solid_particles(n,dx,dy,w_solid,h_solid)
#    print len(x)
#    print len(y)

    plt.figure()
    plt.plot(x,y,'.')

    plt.figure()
    plt.plot(x1,y1,'.')

#test_create_solid_particles()


def test_create_all_particles(n=3, dx = 0.12, dy = 0.12, w_solid = 4.0,
                              h_solid = 4.0, w_fluid = 1.0, h_fluid = 2.0):

    x,y,N_solid,N_fluid = create_all_particles(n,dx,dy,w_solid,h_solid,
                                               w_fluid,h_fluid)

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

    ax.plot_surface(x, y, z, rstride=10, cstride=10, linewidth=0,
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

