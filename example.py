"""
Example code for calculating a molecules electrostatic potential
"""
from qmtools import BasisSet, Molecule, QMTools, Grid
import numpy
import matplotlib.pyplot as plt

# create a calculator object
calculator = QMTools()
basisset = BasisSet("cc-pvdz.bin")

# load the molecule(s), 
# BCB: "./molecule_29766_0/"
# Benzene: "./molecule_241_0/" 
# Hydrogen: "./molecule_783_0/"
# Water: "./molecule_962_0/"

folder1 = "./molecule_783_0/" # Hydrogen
# folder2 = "./molecule_241_0/" # Benzene
mol1 = Molecule(folder1+"GEOM-B3LYP.xyz", folder1+"D-CCSD.npy", basisset)
# mol2 = Molecule(folder2+"GEOM-B3LYP.xyz", folder2+"D-CCSD.npy", basisset)

# parameters
# step: grid step in ANGSTROM
# fat: 'fat' empty space around the molecule in ANGSTROM
# D: convolution constant
# Nd: number of convolutions
step = 0.05
fat = 3.0
D = 0.01
Nd = 20

# create a grid for the density
egrid1 = Grid.DensityGrid(mol1, step, fat)

# calculate the density to the grid and to a numpy 3D array
print(f"Calculating electron density with step {step}")
density1 = calculator.ComputeDensity(mol1, egrid1).reshape((egrid1.shape.x,egrid1.shape.y,egrid1.shape.z), order="F")
print()

# or read the density
# egrid2, density2 = calculator.ReadDensity("./densities/Benzene/Benzene_density_0.05.bin")
# density2 = density2.reshape((egrid2.shape.x, egrid2.shape.y, egrid2.shape.z), order="F")

# create a similar but empty grid for the potential
vgrid1 = egrid1.CopyGrid()
# vgrid2 = egrid2.CopyGrid()

# calculate potential without padding
potential1 = calculator.ComputePotential(mol1, egrid1, vgrid1, spread="gauss").reshape((egrid1.shape.x,egrid1.shape.y,egrid1.shape.z), order="F")

# or with padding
# target = (egrid2.shape.x*2, egrid2.shape.y*2, egrid2.shape.z*2)
# potential2 = calculator.ComputePotential_padded(mol2, egrid2, vgrid2, target_size=target, conv_const=D, conv_count=Nd, spread="conv").reshape((egrid2.shape.x,egrid2.shape.y,egrid2.shape.z), order="F")

# save the results
calculator.WriteGrid_bin(mol1, vgrid1, "962_potential_"+str(step)+".bin")
# calculator.WriteGrid_bin(mol2, vgrid2, "241_potential_"+str(step)+".bin")

# show the results
z_height = 0.0
ind1 = int((z_height - vgrid1.origin.z*0.529772) / step)
# ind2 = int((z_height - vgrid2.origin.z*0.529772) / step)

e1 = density1[:,:,ind1]
p1 = potential1[:,:,ind1]

# e2 = density2[:,:,ind2]
# p2 = potential2[:,:,ind2]

fig, axs = plt.subplots(1,2)
fig.suptitle("Example")

axs[0].imshow(e1, vmin=0, vmax=0.001, origin="lower")
axs[1].imshow(p1, origin="lower")
# axs[1,0].imshow(e2, vmin=0, vmax=0.001, origin="lower")
# axs[1,1].imshow(p2, origin="lower")

fig.savefig("example.png", dpi=400)

del mol1
# del mol2
del calculator
