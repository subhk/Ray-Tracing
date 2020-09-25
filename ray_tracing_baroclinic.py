"""
Ray tracing code
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap, BoundaryNorm
#import matplotlib.colors as colors
#import matplotlib.dates as mdates
from scipy.special import erf

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def k_vec(k, l, m):
    """
    return the resultant wavenumber
    """
    return k*k + l*l + m*m
    
def kH_sq(k, l):
    """
    return the resultant horizontal wavenumber
    """    
    return k**2 + l**2

def N_dist(x,z):
    I = -1.*0.01*np.exp(0.001*z)*erf(x/1.e4)*np.sqrt(np.pi)
    
    return 0.01*np.exp(0.002*z)*(1. + I)
    
def Nz_dist(x,z):
    I = -1.*0.01*np.exp(0.001*z)*erf(x/1.e4)*np.sqrt(np.pi)
    
    return 0.01*0.002*np.exp(0.002*z)*(1. + I)             

def Nx_dist(x,z): 
    Iprime = -1.*2.e-6*np.exp(0.001*z)*np.exp(-1.e-8*x*x)
    
    return 0.01*np.exp(0.002*z)*Iprime
    
    
def V_dist(x, z):
    return -0.2*np.exp(-1.e-8*x*x)*np.exp(0.005*z)
    
    
def Vx_dist(x, z):
    Vx = 4.e-9*x*np.exp(-1.e-8*x*x)*np.exp(0.005*z)
    
    return 0., Vx

def Vy_dist(x, z):    
    return 0., 0.
        
def Vz_dist(x, z):
    Vz = -0.2*0.005*np.exp(-1.e-8*x*x)*np.exp(0.005*z)
    
    return 0., Vz
    
def Vx2_dist(x, z):
    Vx2 = ( 4.e-9 - 8.e-17*x*x )*np.exp(-1.e-8*x*x)*np.exp(0.005*z)        
    
    return 0., Vx2

def Vy2_dist(x, z):
    return 0., 0.
    
def Vz2_dist(x, z):
    Vz2 = -0.2*0.005*0.005*np.exp(-1.e-8*x*x)*np.exp(0.005*z)
    
    return 0., Vz2

def Vxy_dist(x, z):
    return 0., 0.

def Vxz_dist(x, z):
    Vxz = 4.e-9*0.005*x*np.exp(-1.e-8*x*x)*np.exp(0.005*z)
    
    return 0., Vxz    

def Vyz_dist(x, z):
    return 0., 0.
    

def Dispersion_(f, N2, k, l, m):
    """
    WKB dispersion relation: ω = K(k,l,m)
    """    
    
    kH2 = kH_sq(k, l)
    
    
    ω = f + 0.5*(Vx-Uy) + 0.5*N2*kH2/(f*m**2) + (Uz*l-Vz*k)/m 
    ω = ω + k*U + l*V
                    
    return ω

    
def GroupVel_(k, l, m, N2, f, U, V, x, z):
    """
    group velocity
    """    
    
    kH2 = kH_sq(k, l)
    
    Uz,  Vz = Vz_dist(x, z)
    
    Cg_x = N2*k/(f*m**2.) - Vz/m
    Cg_z = -N2*kH2/(f*m**3.) - 1./m**2*(Uz*l - Vz*k)
    
    return Cg_x, Cg_z
   
        
def Dk(k, l, m, N2, Nx, Nz, f, x, z):    
    """
    change in k with time
    """
    
    N = np.sqrt(N2)
    kH2 = kH_sq(k, l)
    
    Ux,  Vx  = Vx_dist (x, z)
    Ux2, Vx2 = Vx2_dist(x, z)
    Uxy, Vxy = Vxy_dist(x, z)
    Uxz, Vxz = Vxz_dist(x, z)
    
    term1 = 0.5*( Vx2 - Uxy )
    term2 = N*Nx*kH2/(f*m**2.)
    term3 = ( Uxz*l - Vxz*k )/m
    term4 = k*Ux + l*Vx
    
    return -1.*(term1 + term2 + term3 + term4)
    
    
def Dm(k, l, m, N2, Nx, Nz, f, x, z):    
    """
    change in m with time
    """
    
    N = np.sqrt(N2)
    kH2 = kH_sq(k, l)
    
    Uz,  Vz  = Vz_dist (x, z)
    Uz2, Vz2 = Vz2_dist(x, z)
    Uyz, Vyz = Vyz_dist(x, z)
    Uxz, Vxz = Vxz_dist(x, z)
    
    term1 = 0.5*( Vxz - Uyz )
    term2 = N*Nz*kH2/(f*m**2.)
    term3 = ( Uz2*l - Vz2*k )/m
    term4 = k*Uz + l*Vz
    
    return -1.*(term1 + term2 + term3 + term4)


def RK44_timestep(k, l, m, N2, Nx, Nz, f, x0, z0, dt):

    U = 0.
    
    Cg_x1, Cg_z1 = GroupVel_(k, l, m, N2, f, U, V_dist(x0, z0), x0, z0)
    Δωx_1 = Dk(k, l, m, N2, Nx, Nz, f, x0, z0)
    Δωz_1 = Dm(k, l, m, N2, Nx, Nz, f, x0, z0)
     
    ( xnew, znew ) = ( x0 + 0.5*dt*Cg_x1, z0 + 0.5*dt*Cg_z1 )
    ( knew, lnew, mnew ) = ( k + 0.5*dt*Δωx_1, l, m + 0.5*dt*Δωz_1 )    
    Cg_x2, Cg_z2 = GroupVel_(knew, lnew, mnew, N2, f, U, V_dist(xnew, znew), xnew, znew)
    Δωx_2 = Dk(knew, lnew, mnew, N2, Nx, Nz, f, xnew, znew)
    Δωz_2 = Dm(knew, lnew, mnew, N2, Nx, Nz, f, xnew, znew)

    ( xnew, znew ) = ( xnew + 0.5*dt*Cg_x2, znew + 0.5*dt*Cg_z2 )
    ( knew, lnew, mnew ) = ( knew + 0.5*dt*Δωx_2, lnew, mnew + 0.5*dt*Δωz_2 )
    Cg_x3, Cg_z3 = GroupVel_(knew, lnew, mnew, N2, f, U, V_dist(xnew, znew), xnew, znew)
    Δωx_3 = Dk(knew, lnew, mnew, N2, Nx, Nz, f, xnew, znew)
    Δωz_3 = Dm(knew, lnew, mnew, N2, Nx, Nz, f, xnew, znew)

    ( xnew, znew ) = ( xnew + dt*Cg_x3, znew + dt*Cg_z3 )
    ( knew, lnew, mnew ) = ( knew + dt*Δωx_3, lnew, mnew + dt*Δωz_3 )
    Cg_x4, Cg_z4 = GroupVel_(knew, lnew, mnew, N2, f, U, V_dist(xnew, znew), xnew, znew)
    Δωx_4 = Dk(knew, lnew, mnew, N2, Nx, Nz, f, xnew, znew)
    Δωz_4 = Dm(knew, lnew, mnew, N2, Nx, Nz, f, xnew, znew)    
        
    x0 = x0 + dt/6.*( Cg_x1 + 2.*Cg_x2 + 2.*Cg_x3 + Cg_x4 ) 
    z0 = z0 + dt/6.*( Cg_z1 + 2.*Cg_z2 + 2.*Cg_z3 + Cg_z4 )
    
    k =  k + dt/6.*( Δωx_1 + 2.*Δωx_2 + 2.*Δωx_3 + Δωx_4 )
    l =  l + 0.
    m =  m + dt/6.*( Δωz_1 + 2.*Δωz_2 + 2.*Δωz_3 + Δωz_4 )
    
    return x0, z0, k, l, m
       

def ray_tracing():

    # depth of the domain
    H = -500.
    
    # initial origin of the ray
    x0 = [-20.e3, -10.e3, 0., 5.e3, 10.e3]
    z0 = 0.
    
    # initial jet structure (barotropic)
#    U = 0.
#    V = -0.2*np.exp(-1.e-8*x0**2)
        
    # Coriolis frequency
    f = 1.e-4
    
    # initial wavelengths
    λx = 40.e3 # horizontal wavelength
    λz = 100.   # vertical wavelength
    
    # initial wavenumbers
    k0 = 2.*np.pi/λx
    l0 = 0.
    m0 = 2.*np.pi/λz
    
    save_rate = 500         # file save rate
    tot_iter = 4000000       # total iteration
    
    xp = np.zeros( ( len(x0), int(tot_iter/save_rate) ) )
    zp = np.zeros( ( len(x0), int(tot_iter/save_rate) ) )
    
    sze = np.zeros( len(x0) ) 
    
    for xt in range( len(x0) ):
        
        xp[xt,0] = x0[xt]
        zp[xt,0] = z0
            
    # simulation time-step (in sec.)
    dt = 20.
    
    for jt in range( len(x0) ):
    
        cnt = 0
        ( x_, z_ ) = ( xp[jt,0], zp[jt,0] )
        k = k0
        l = l0
        m = m0
          
        for it in range(1, tot_iter):
        
            #if it == 1: ( x_, z_ ) = ( xp[jt,it-1], zp[jt,it-1] )

            N  = N_dist (x_, z_)            
            Nx = Nx_dist(x_, z_)
            Nz = Nz_dist(x_, z_)                    
            x_, z_, k, l, m = RK44_timestep(k, l, m, N*N, Nx, Nz, f, x_, z_, dt)
            
            if it%save_rate == 0:
                cnt += 1 
                # store the path data    
                xp[jt,int(it/save_rate)] = x_
                zp[jt,int(it/save_rate)] = z_
                sze[jt] = int(it/save_rate)
                print('iteration = %i %f' %(it, m))
            
            if z_ < H:    
                print('wave reached sea-floor')
                break
                
            if z_ > 0:   
                print('wave reached sea-surface')
                break
                
    xg = np.linspace(-40e3, 40e3, 500)
    zg = np.linspace(H, 0, 100)
    Xg, Zg = np.meshgrid(xg, zg, indexing='ij') 
    Vg = -0.2*np.exp(-1.e-8*Xg**2)*np.exp(0.005*zg)            
    
    fig, ax = plt.subplots()
    cf = ax.contour( Xg/1.e3, Zg/1.e3, Vg, levels=5, colors='black')
    
    
    for it in range( len(x0) ):
        print('value = ', xp[it,0], zp[it,0])
        ax.plot(xp[it, 0:int(sze[it])]/1.e3, zp[it, 0:int(sze[it])]/1.e3, '-b')
        
#    ax.plot(xp[0,:]/1.e3, zp[0,:]/1.e3, '-b')        
    
    plt.xlim([np.min(xg)/1.e3, np.max(xg)/1.e3])
    plt.ylim([np.min(zg)/1.e3, np.max(zg)/1.e3])
    
    plt.xlabel(r'$x$(km)', fontsize=14)
    plt.ylabel(r'$z$(km)', fontsize=14)    
    plt.grid(True)
    plt.show()
    
    
if __name__ == "__main__":

    ray_tracing()
    
             
