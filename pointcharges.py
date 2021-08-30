"""
Script to test the implemented FFT solution with point charges (no spreading)
Size of the test grid can be adjusted with the variable 'target',
positions of the point charges are printed when the script is run, by default they're placed
1/4 of the grid from the middle.
"""
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
folder = "./molecule_783_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset) 

step = 0.05
fat = 3.0

egrid = Grid.DensityGrid(mol, step, fat)
dims = [egrid.shape.x, egrid.shape.y, egrid.shape.z]
density = calculator.ComputeDensity(mol, egrid).reshape((dims[0], dims[1], dims[2]), order="F")

# use this to add padding
target = (egrid.shape.x, egrid.shape.y, egrid.shape.z)

# test
vgrid = egrid.CopyGrid()
potential = calculator.ComputePotential_padded(mol, egrid, vgrid, target_size=target, spread="test").reshape((vgrid.shape.x,vgrid.shape.y,vgrid.shape.z), order="F")

# plot
plt.plot(range(potential.shape[0]), potential[:, int(dims[1]/2), int(dims[2]/2)])
plt.title("Test charges, step="+str(step)+" no padding")
plt.ylabel("[V]")
plt.savefig("points.png")
