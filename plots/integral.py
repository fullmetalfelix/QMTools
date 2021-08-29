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
folder = "../molecule_29766_0/"
step = 0.05
fat = 3.0
D = 0.01
Nd = 20

# grid
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder +"D-CCSD.npy", basisset)
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
plt.plot(range(dims[0]), p_g[:, int(dims[1]/2), int(dims[2]/2)], label="Gaussian")
print()

# potential, conv
vgrid = egrid.CopyGrid()
print(f"Calculating {molecule} potential with step {step} and convolution constants Nd={Nd}, D={D}")
p_c = calculator.ComputePotential(mol, egrid, vgrid, conv_count=Nd, conv_const=D, spread="conv").reshape((dims[0], dims[1], dims[2]), order="F")
plt.plot(range(dims[0]), p_c[:, int(dims[1]/2), int(dims[2]/2)], label="Convolution; "+str(Nd)+"x"+str(D))
print()

# potential, integral
vgrid = egrid.CopyGrid()
print(f"Calculating {molecule} potential with step {step} by integrating")
p_i = calculator.ComputeHartree(mol, egrid, vgrid).reshape((dims[0], dims[1], dims[2]), order="F")
plt.plot(range(dims[0]), p_i[:, int(dims[1]/2), int(dims[2]/2)], label="Integral")
print()

# plot
# nuclei
for nuclei in range(mol.natoms):
  if (mol.coords[nuclei].y-vgrid.origin.y)/(step*1.8897259886) >= int(dims[1]/2)-5 and (mol.coords[nuclei].y-vgrid.origin.y)/(step*1.8897259886) <= int(dims[1]/2)+5:
    plt.scatter((mol.coords[nuclei].x-vgrid.origin.x)/(step*1.8897259886), 0.0, color=chrg[mol.types[nuclei]])

#properties
plt.title(molecule+" method comparison, step="+str(step))
plt.legend()
plt.savefig("methods_"+str(step)+".png", dpi=400)

del mol
del calculator
