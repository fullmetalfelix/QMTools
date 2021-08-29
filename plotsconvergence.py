import glob
from qmtools import BasisSet, Molecule, QMTools, Grid
import numpy
import time
import matplotlib.pyplot as plt

# plot stuff
plt.style.use('ggplot')
chrg = {1:"palevioletred", 6:"saddlebrown", 8:"red",35:"darkgoldenrod"}

# create a calculator object
calculator = QMTools()
basisset = BasisSet("cc-pvdz.bin")

# params.
molecule = "BCB"
folder = "./molecule_29766_0/"
fat = 3.0
D = 0.01
Nd = 20

# grid and plot init.
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder +"D-CCSD.npy", basisset)
plt.rcParams['figure.constrained_layout.use'] = True
fig, axs = plt.subplots(1,3, figsize=(8,4))
fig.suptitle(molecule+", step convergence using Gaussian densities") 
col = 0

for step in[0.1, 0.05, 0.01]:
  egrid = Grid.DensityGrid(mol, step, fat)
  dims = [egrid.shape.x, egrid.shape.y, egrid.shape.z]

  # density
  print(f"Calculating {molecule} density with step {step}")
  density = calculator.ComputeDensity(mol1, egrid).reshape((dims[0], dims[1], dims[2]), order="F")
  print()
  
  # potential, gauss
  vgrid = egrid.CopyGrid()
  print(f"Calculating {molecule} potential with step {step} and Gaussian densities")
  p_g = calculator.ComputePotential(mol, egrid, vgrid, spread="gauss").reshape((dims[0], dims[1], dims[2]), order="F")
  axs[col].plot(range(dims[0]), p_g[:, int(dims[1]/2), int(dims[2]/2)], label=str(step))
  print()
  
  # plot current step
  print(f"Plotting {molecule}")
  axs[col].set_title(str(step)+" Å")
  axs[col].set_ylabel("Electrostatic potential [V]")
  axs[col].set_xlabel("Position in grid [Å/step]")
  for nuclei in range(mol.natoms):
    if (mol.coords[nuclei].y-vgrid.origin.y)/(step*1.8897259886) >= int(dims[1]/2)-5 and (mol.coords[nuclei].y-vgrid.origin.y)/(step*1.8897259886) <= int(dims[1]/2)+5:
      axs[col].scatter((mol.coords[nuclei].x-vgrid.origin.x)/(step*1.8897259886), 0.0, color=chrg[mol.types[nuclei]])
  col+=1
  
plt.savefig("convergence_gauss.png", dpi=400)

del mol
del calculator
