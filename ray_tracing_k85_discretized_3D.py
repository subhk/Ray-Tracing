import math
import numpy as np
import R_tools_new_goth as tN
import sys
import time
from netCDF4 import Dataset
from scipy import integrate
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

def gradients_z_ddx_ddz(T, x, z, dn=2):
  dx = x[dn:]-x[:-dn]
  dz = z[dn:]-z[:-dn]
  [nx,nz] = T.shape

  Tx_r = np.zeros((nx,nz))
  Tz_z = np.zeros((nx,nz))

  Tx_r[dn-1:-(dn-1),:] = (T[dn:,:]-T[:-dn,:])/dx[:,None]
  Tz_z[:,dn-1:-(dn-1)] = (T[:,dn:]-T[:,:-dn])/dz[None,:]
  Tx_r[0,:]  = Tx_r[1,:]
  Tx_r[-1,:] = Tx_r[-2,:]
  Tz_z[:,0]  = Tz_z[:,1]
  Tz_z[:,-1] = Tz_z[:,-2]
  return Tx_r, Tz_z

start_time = time.time()
       
dr = '/nautilus/luwei/niskin2km_sm/wind_impulse_his/k85_flow/'
fn = 'k85_baroclinic.nc'
nc = Dataset(dr + fn, 'r')
x  = nc.variables['x'][:]
z  = nc.variables['z'][::-1]
U  = 0.
f  = 1.e-4
beta  = 0
V_xz  = tN.ncload(nc, 'V')[:,::-1]
N2_xz = tN.ncload(nc, 'N2')[:,::-1]

Vx_xz,  Vz_xz  = gradients_z_ddx_ddz(V_xz,  x, z)
N2x_xz, N2z_xz = gradients_z_ddx_ddz(N2_xz, x, z)
Vxx_xz, Vxz_xz = gradients_z_ddx_ddz(Vx_xz, x, z)
Vzx_xz, Vzz_xz = gradients_z_ddx_ddz(Vz_xz, x, z)

U = Ux = Uy = Uz = Uyx = Uyy = Uyz = Uzx = Uzy = Uzz = 0
Vy = Vxy = Vzy = 0

# interpolation
varlist = ['N2',  'V',
           'N2x', 'N2z',
           'Vx',  'Vz',
           'Vxx', 'Vxz',
           'Vzx', 'Vzz']

varfun_d = {}
for var in varlist:
    varfun_d[var] = RegularGridInterpolator((x, z), globals()[var+'_xz'][:])

def Baroclinic(x, z):
    x_interp = np.array([x])
    z_interp = np.array([z])
    #print 'x, z = ', x_interp, z_interp
    new_points_xz = np.array([x, z])
    for I, var in enumerate(varlist):
        globals()[var] = varfun_d[var](new_points_xz)[0]
    f_eff  = f + 0.5 * Vx
    #print 'Variables for the new point are calculated!'
    U = Ux = Uy = Uz = Uyx = Uyy = Uyz = Uzx = Uzy = Uzz = 0
    Vy = Vxy = Vzy = 0
    N2y   = 0  
    return f_eff, N2, U, V, Ux, Uy, Uz, Vx, Vy, Vz, N2x, N2y, N2z, Vxx, Vxy, Vxz, Uyx, Uyy, Uyz, Vzx, Vzy, Vzz, Uzx, Uzy, Uzz

class RayTracing():
    
    def __init__(self, t0=0, t_final=24*365, N_eval=24*365+1):
        self.t0 = t0 * 3600
        self.t_final = t_final * 3600
        self.t_eval = np.linspace(self.t0, self.t_final, N_eval)

    def odeparams(self, t, var):
        x, y, z, kx, ky, kz = var
        #print 't = ', t, ',    x, y, z = ', x, y, z
        f_eff, N2, U, V, Ux, Uy, Uz, Vx, Vy, Vz, N2x, N2y, N2z, Vxx, Vxy, Vxz, Uyx, Uyy, Uyz, Vzx, Vzy, Vzz, Uzx, Uzy, Uzz = Baroclinic(x, z)
        cgx  = N2 * kx / f / kz ** 2 - Vz / kz + U 
        cgy  = N2 * ky / f / kz ** 2 + Uz / kz + V   
        cgz  = - N2 * (kx ** 2 + ky ** 2) / f / kz ** 3 - Uz * ky / kz ** 2 + Vz * kx / kz ** 2
        rhsx = - (0.5 * Vxx - 0.5 * Uyx + N2x * (kx ** 2 + ky ** 2) / f / 2 / kz ** 2 + Uzx * ky / kz - Vzx * kx / kz + kx * Ux + ky * Vx) 
        rhsy = - (beta + 0.5 * Vxy - 0.5 * Uyy + N2y * (kx ** 2 + ky ** 2) / f / 2 / kz ** 2 + Uzy * ky / kz - Vzy * kx / kz + kx * Uy + ky * Vy) 
        rhsz = - (0.5 * Vxz - 0.5 * Uyz + N2z * (kx ** 2 + ky ** 2) / f / 2 / kz ** 2 + Uzz * ky / kz - Vzz * kx / kz + kx * Uz + ky * Vz)
        return cgx, cgy, cgz, rhsx, rhsy, rhsz

    def omega(self, x, y, z, kx, ky, kz):
        f_eff, N2, U, V, Ux, Uy, Uz, Vx, Vy, Vz, N2x, N2y, N2z, Vxx, Vxy, Vxz, Uyx, Uyy, Uyz, Vzx, Vzy, Vzz, Uzx, Uzy, Uzz = Baroclinic(x, z)
        # dispersion relation (should be a constant)
        omega   = f_eff + N2 * (kx ** 2 + ky ** 2) / 2 / f / kz ** 2 + Uz * ky / kz - Vz * kx / kz + kx * U + ky * V
        # intrinsic frequency
        omega_0 = f_eff + N2 * (kx ** 2 + ky ** 2) / 2 / f / kz ** 2 + Uz * ky / kz - Vz * kx / kz
        return omega, omega_0

    def output_line(self, t, x, y, z, Lx, Ly, Lz, var, omega, omega_0):
        Cgx, Cgy, Cgz, rhsx, rhsy, rhsz = var
        output_line = '{0:8.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:0.2e}\t{5:0.2e}\t{6:0.2e}\t{7:.2e}\t{8:.2e}\t{9:.2e}\t{10:.4e}\t{11:.4e}\n'.format(t/3600, x, y, z, Lx, Ly, Lz, Cgx, Cgy, Cgz, omega, omega_0)
        return output_line

    def open_data_file(self):
        self.filename = 'ray_data_k85_1yr_bc_' + '{:.0f}'.format(x0/1000) + '_{:.0f}'.format(theta) + '_interp_3D_5min.out' 
        print 'filename: ', self.filename
        file_output = open(dr + self.filename, 'w')
        file_output.write('time [hr]\tx\ty\tz\tL_x\t\tL_y\t\tL_z\t\tCg_x\t\tCg_y\t\tCg_z\t\tomega\t\tomega_0\n')
        return file_output

    def run_simulation(self, x0, y0, z0, lx0, lz0, theta, x_lim=[-50e3, 50e3], z_lim=-0.5e3):
        """
        uses integrate.solve_ivp, with method='rk45' to solve the differential equations.
        """
        theta = np.deg2rad(theta)
        kx0   = 2 * np.pi / lx0 * np.cos(theta)
        ky0   = 2 * np.pi / lx0 * np.sin(theta)
        kz0   = 2 * np.pi / lz0

        # defining space limits for the rk45 algorithm
        def z_wall1(t, y): return y[2] - z_lim
        z_wall1.terminal = True
        def x_wall1(t, y): return y[0] - x_lim[0]
        x_wall1.terminal = True
        def x_wall2(t, y): return y[0] - x_lim[1]
        x_wall2.terminal = True

        # RK45 algorithm
        sol = integrate.solve_ivp(fun=self.odeparams, t_span=(self.t0, self.t_final), y0=(x0, y0, z0, kx0, ky0, kz0), method='RK45', t_eval=self.t_eval, events=[x_wall1, x_wall2, z_wall1], max_step=300)

        # write to output file
        file_output = self.open_data_file()
        for i in range(len(sol.t)):
            x, y, z, kx, ky, kz = sol.y[:, i]
            #print sol.t[i]/3600,sol.y[:,i] 
            Lx = 2 * np.pi / kx
            Ly = 2 * np.pi / ky
            Lz = 2 * np.pi / kz
            omega, omega_0 = self.omega(x, y, z, kx, ky, kz)
            var = self.odeparams(sol.t[i], sol.y[:,i])
            #print var
            file_output.write(self.output_line(sol.t[i], x, y, z, Lx, Ly, Lz, var, omega, omega_0))
        file_output.close()
        return x, y, z, kx, ky, kz


if __name__ == "__main__":

    #coordinates = [(x * 1e3, 0, 0) for x in np.linspace(-50, 50, 21)]  
    coordinates = [(-20 * 1e3, 0, 0)]  

    # simulation
    ray_system = RayTracing()
    Lx0, Lz0, theta = 40e+3, 1e+2, 60 
    for x0, y0, z0 in coordinates: 
        print 'ray tracing starts here... plotting skipped'
        print 'x0 = ', x0/1e+3, 'km, z0 = ', z0, 'm' 
        ray_system.run_simulation(x0, y0, z0, Lx0, Lz0, theta, x_lim=[-60e3, 60e3], z_lim=-600)

print("--- %s seconds ---" % (time.time() - start_time))
