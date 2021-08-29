
import time
import struct
import numpy as np
import matplotlib.pyplot as plt

from poisson import *

# Load data
rho_el, mol_xyz, grid_origin, grid_vec, step = read_density_bin('density_0.025.bin')
rho_el = -rho_el*(1/step)**3 # From e/voxel to -e/ang^3

# Extend box
target_size = (240,240,128)
pad_size = [(target_size[i]-rho_el.shape[i]) // 2 for i in range(3)]
pad_size = [(p,p) for p in pad_size]
rho_el = np.pad(rho_el, pad_size, mode='constant')
grid_origin = [grid_origin[i]-grid_vec[i][i]*pad_size[i][0] for i in range(3)]
scan_window = (
    grid_origin,
    tuple([grid_origin[i] + grid_vec[i][i]*rho_el.shape[i] for i in range(3)])
)

# Add nuclear density to total density
rho_nuclei = nuclear_density(mol_xyz, sigma=step, scan_dim=rho_el.shape, scan_window=scan_window)
rho_total = rho_el + rho_nuclei

# Solve for potential via Poisson equation
t0 = time.time()
pot_total_fft = poisson_fft(rho_total, scan_window=scan_window)
print(f'Poisson solution time: {time.time()-t0:.4f}')

save_cube(pot_total_fft, mol_xyz, grid_origin, grid_vec, file_path='pot_extend.cube')

# Plot slice of potential
z_height = 1
ind = int((z_height - grid_origin[2]) / step)
pot_slice = pot_total_fft[:,:,ind]

fig = plt.figure(figsize=(6, 5))
plt.imshow(pot_slice.T, origin='lower')
plt.title('Coulomb potential')
plt.colorbar()

plt.savefig('pot_extend.png')
