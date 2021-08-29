import glob
import sys
from qmtools import BasisSet, Molecule, QMTools, Grid
import numpy
from ctypes import c_float, c_int
import time
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# create a calculator object
calculator = QMTools()
basisset = BasisSet("cc-pvdz.bin")

# this is just to create a grid
molecule = "Hydrogen"
folder = "../molecule_783_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset) 

step = 0.02
fat = 3.0

egrid = Grid.DensityGrid(mol, step, fat)
density = calculator.ComputeDensity(mol, egrid).reshape((dims[0], dims[1], dims[2]), order="F")
dims = [egrid.shape.x, egrid.shape.y, egrid.shape.z]

# use this to add padding
target = (egrid.shape.x, egrid.shape.y, egrid.shape.z)

vgrid = egrid.CopyGrid()
potential = calculator.ComputePotential_padded(mol, egrid, vgrid, target_size=target, spread="test").reshape((vgrid.shape.x,vgrid.shape.y,vgrid.shape.z), order="F")
plt.plot(range(potential.shape[0]), potential[:, int(dims[1]/2), int(dims[2]/2)])
plt.savefig("points.png")
