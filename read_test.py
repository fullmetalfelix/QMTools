import glob
from qmtools import BasisSet, Molecule, QMTools, Grid
import numpy
from ctypes import c_float
import time
import struct


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

#vgrid = Grid.DensityGrid_load(mol, "pot_0.025.bin")
#egrid = calculator.ReadDensity("density_0.025.bin", 0.05)
print()
vgrid = calculator.ReadDensity("pot_0.05.bin")
print(vgrid)

#calculator.ComputePotential(mol, vgrid)

calculator.WriteGrid_xsf(mol, vgrid, "pot_test.xsf")