
import time
import struct
import numpy as np
import matplotlib.pyplot as plt

R_BOHR = 0.529772 # Bohr-radius in angstroms
HARTREE = 27.211386245 # eV
COULOMB_CONST = 14.399644 # [eV*Ang/e^2]
EPS0 = 0.0055263494 # e/(V*Ang)

def read_density_bin(file_path):

    with open(file_path, 'rb') as f:

        # Atom coordinates
        N = struct.unpack('i', f.read(4))[0]
        Zs = struct.unpack('i'*N, f.read(4*N))
        mol_xyz = struct.unpack('f'*N*3, f.read(4*N*3))
        mol_xyz = np.array(mol_xyz).reshape(N, 3) * R_BOHR
        mol_xyz = np.concatenate([mol_xyz.T, [Zs]]).T
        # Grid coordinates
        grid_origin = np.array(struct.unpack('fff', f.read(4*3))) * R_BOHR
        grid_shape = struct.unpack('iii', f.read(4*3))
        N_grid = struct.unpack('i', f.read(4))[0]
        
        assert N_grid == grid_shape[0]*grid_shape[1]*grid_shape[2]
        step = struct.unpack('f', f.read(4))[0]*R_BOHR
        grid_vec = np.array([
            [step, 0.0, 0.0],
            [0.0, step, 0.0],
            [0.0, 0.0, step]
        ])

        # Density
        density = np.frombuffer(f.read(), dtype=np.float32).reshape(grid_shape, order='F')

    return density, mol_xyz, grid_origin, grid_vec, step

def poisson_fft(rho, scan_window=((-10, -10, -10), (10, 10, 10))):
    ndim = rho.ndim
    L = [scan_window[1][i]-scan_window[0][i] for i in range(ndim )]
    N = rho.shape
    rho=-rho/EPS0
    omega2 = [-(2*np.pi*np.fft.fftfreq(N[i], L[i]/N[i])) ** 2 for i in range(ndim)]
    omega2 = np.stack(np.meshgrid(*omega2, indexing='ij'), axis=-1).sum(axis=-1).astype(np.float32)
    rho_hat = np.fft.fftn(rho)
    phi_hat = rho_hat / omega2
    phi_hat[(0,)*ndim] = 0
    pot = np.real(np.fft.ifftn(phi_hat))
    return pot

def save_cube(density, mol_xyz, grid_origin, grid_vec, file_path='hartree.cube'):
    N = len(mol_xyz)
    grid_shape = density.shape
    mol_xyz = mol_xyz.copy()
    with open(file_path, 'w') as f:
        f.write('Comment line\nComment line\n')
        f.write(f'{N} {" ".join([str(o/R_BOHR) for o in grid_origin])}\n')
        for i in range(3):
            f.write(f'{grid_shape[i]} {" ".join([str(v) for v in grid_vec[i]/R_BOHR])}\n')
        for x in mol_xyz:
            x[:3] /= R_BOHR
            f.write(f'{int(x[-1])} 0.0 {x[0]} {x[1]} {x[2]}\n')
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    f.write(f'{density[i, j, k]} ')
                    if (k+1) % 6 == 0:
                        f.write('\n')
                f.write('\n')

def nuclear_density(mol_xyz, sigma=0.5, scan_dim=(200,200,200), scan_window=((-10, -10, -10), (10, 10, 10))):
    r_grid = [np.linspace(scan_window[0][i], scan_window[1][i], scan_dim[i]) for i in range(3)]
    r_grid = np.stack(np.meshgrid(*r_grid, indexing='ij'), axis=3)
    rho = np.zeros(scan_dim)
    for mol in mol_xyz:
        dr = np.linalg.norm(r_grid-mol[:3], axis=3)
        rho += mol[-1]/(sigma*np.sqrt(2*np.pi))**3 * np.exp(-dr**2 / (2*sigma**2))
    return rho

if __name__ == '__main__':

    # Load electron density from binary file
    rho_el, mol_xyz, grid_origin, grid_vec, step = read_density_bin('density_0.025.bin')
    rho_el = -rho_el*(1/step)**3 # From e/voxel to e/ang^3

    #scan_window = ((from), (to))
    scan_window = (
        grid_origin,
        tuple([grid_origin[i] + grid_vec[i][i]*rho_el.shape[i] for i in range(3)])
    )

    t0 = time.time()
    # Add nuclear density to total density
    rho_nuclei = nuclear_density(mol_xyz, sigma=step, scan_dim=rho_el.shape, scan_window=scan_window)
    rho_total = rho_el + rho_nuclei

    # Solve for potential via Poisson equation
    
    pot_total_fft = poisson_fft(rho_total, scan_window=scan_window)
    print(f'Poisson solution time: {time.time()-t0:.4f}')

    save_cube(pot_total_fft, mol_xyz, grid_origin, grid_vec, file_path='pot.cube')

    # Plot slice of potential
    z_height = 1
    ind = int((z_height - grid_origin[2]) / step)
    pot_slice = pot_total_fft[:,:,ind]

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(pot_slice.T, origin='lower')
    plt.title('Coulomb potential')
    plt.colorbar()

    plt.savefig('pot.png')
