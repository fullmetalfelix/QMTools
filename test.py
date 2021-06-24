import glob
from qmtools import BasisSet, Molecule, QMTools, Grid
import numpy
from ctypes import c_float
import time


# create a calculator object
calculator = QMTools()
basisset = BasisSet("cc-pvdz.bin")


# load the molecule
folder = "./molecule_29766_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset)
# the xyz must contain atomic coordinates in ANGSTROM



# create a grid for the density
# mol: the molecule
# 0.1: grid step in ANGSTROM
# 3.0: 'fat' empty space around the molecule in ANGSTROM
egrid = Grid.DensityGrid(mol, 0.05, 3.0)
print(egrid)

#vgrid = Grid.MakeGrid([-8,-8,0], 0.1, [160, 160, 64])
#print()
#vgrid = Grid.DensityGrid(mol, 0.1, 3.0)
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

density = calculator.ComputeDensity(mol, egrid)
calculator.WriteDensity(mol, egrid, "density_0.025.bin")

print()
calculator.poisson_fft(mol, egrid)
#calculator.padded_poisson(mol, egrid, target_size=(120,120,64))
print()

#density = calculator.ComputeDensity_subgrid(mol, egrid)
#calculator.WriteDensity(mol, egrid, "density_0.1_sg4.bin")
#t0 = time.time()
#hartree = calculator.ComputeHartree(mol, egrid, vgrid)
#print(f'Hartree solution time: {time.time()-t0:.4f}')
#calculator.WriteDensity(mol, vgrid, "hartree_exp.bin")

del mol
del calculator
