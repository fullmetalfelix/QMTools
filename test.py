import glob
from qmtools import BasisSet, Molecule, QMTools, Grid
import numpy
from ctypes import c_float
import time


# create a calculator object
calculator = QMTools()
basisset = BasisSet("cc-pvdz.bin")


# load the molecule, 
# BCB: "./molecule_29766_0/"
# Benzene: "./molecule_241_0/" 
# Hydrogen: "./molecule_783_0/"
# Water: "./molecule_962_0/"

folder = "./molecule_241_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset)
# the xyz must contain atomic coordinates in ANGSTROM


# - create a grid for the density
# step: grid step in ANGSTROM
# fat: 'fat' empty space around the molecule in ANGSTROM
step = 0.025
fat = 3.0

egrid = Grid.DensityGrid(mol, step, fat)
print(egrid)

density = calculator.ComputeDensity(mol, egrid)
calculator.WriteGrid_bin(mol, egrid, "density_"+str(step)".bin")

#vgrid = Grid.MakeGrid([-8,-8,0], 0.1, [160, 160, 64])
#print()
#vgrid = Grid.DensityGrid(mol, 0.05, 3.0)
#print(vgrid)

#grid = Grid.TestGrid(mol, 0.1, 0)
#print(grid)


'''
grid.qube[1] = c_float(1)
grid.qube[grid.shape.x] = c_float(2)
grid.qube[grid.shape.x*grid.shape.y] = c_float(3)

print(grid._qube[1,0,0])
print(grid._qube[0,1,0])
print(grid._qube[0,0,1])
'''

# - parameters for potential calculations
# target_size: padded grid size, 
# D: convolution constant
# Nd: number of convolutions
# target_size: size of padded grid, if size is kept egrid.shape.{xyz} no padding will be added

D = 0.01
Nd = 20
target_size = (egrid.shape.x, egrid.shape.y, egrid.shape.z)

print()
vgrid = egrid.CopyGrid()
calculator.ComputePotential_padded(mol, egrid, vgrid, target_size=target_size, spread="gauss")
calculator.WriteGrid_bin(mol, egrid, "potential_"+str(step)".bin")



#density = calculator.ComputeDensity_subgrid(mol, egrid)
#calculator.WriteDensity(mol, egrid, "density_0.1_sg4.bin")
# t0 = time.time()
# hartree = calculator.ComputeHartree(mol, egrid, vgrid)
# print(f'Hartree solution time: {time.time()-t0:.4f}')
#calculator.WriteDensity(mol, vgrid, "hartree_exp.bin")

del mol
del calculator
