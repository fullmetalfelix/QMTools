import psi4
import numpy

psi4.core.set_output_file('geoopt.out', False)

psi4.set_memory('24 GB')

mol = psi4.geometry("""
units ang
symmetry c1
#FLAGXYZ
1	0.36 0.0 0.0
1	-0.36 0.0 0.0 
""")

mol.fix_com(True)
mol.fix_orientation(True)


psi4.core.set_num_threads(4)
psi4.set_options({'basis': 'cc-pvdz'})
psi4.set_options({'maxiter': 500})
psi4.set_options({'cachelevel': 1})
psi4.set_options({'reference': 'rhf'})
psi4.set_options({'opt_coordinates': 'cartesian'})

# --- GEO-OPT with B3LYP --- #
E, wf = psi4.optimize('B3LYP', molecule=mol, return_wfn=True)

# extract the geometry in angstroms
xyz = wf.molecule().geometry().to_array(True,True) * 0.529177 # converted to angs

# save the some matrixes
numpy.save("S", wf.S().to_array(False, True))
numpy.save("D-B3LYP", wf.Da().to_array(False, True))
numpy.save("F-B3LYP", wf.Fa().to_array(False, True))
numpy.save("C-B3LYP", wf.Ca().to_array(False, True))
numpy.save("H-B3LYP", wf.H().to_array(False, True))
numpy.save("e-B3LYP", wf.epsilon_a().to_array(False, True))


# save the xyz
fxyz = open("GEOM-B3LYP.xyz", "w")
for i in range(mol.natom()):
	fxyz.write("{:d}\t{} {} {}\n".format(int(mol.Z(i)), xyz[i,0], xyz[i,1], xyz[i,2]))
fxyz.close()


'''
# --- HARTREE-FOCK ENERGY AND WF ---#
E, wf = optimize('scf', return_wfn=True)
print "HF energy:", E
oeprop(wf, 'MULLIKEN_CHARGES', title='MYHFMULLIKEN')

# --- SAVE DENSITY MATRIX and others ---#
SaveMatrix("D-HF.bin", wf.Da().to_array(copy=False, dense=True))
SaveMatrix("F-HF.bin", wf.Fa().to_array(copy=False, dense=True))
SaveMatrix("C-HF.bin", wf.Ca().to_array(copy=False, dense=True))
SaveMatrix("H-HF.bin", wf.H().to_array(copy=False, dense=True))

# --- SAVE OVERLAP MATRIX ---#
SaveMatrix("S.bin", wf.S().to_array(copy=False, dense=True))

# --- CCSD starting from the HFWF ---#
grad, ccwf = gradient('ccsd', return_wfn=True, ref_wfn=wf)
ccwf.gradient().print_out()
print "CCSD energy:", ccwf.energy()

oeprop(ccwf, 'MULLIKEN_CHARGES', title='MYMULLIKEN')
oeprop(ccwf, 'DIPOLE', title='MYDIPOLE')


# --- SAVE DENSITY MATRIX ---#
SaveMatrix("D-CCSD.bin", ccwf.Da().to_array(copy=False, dense=True))

# --- SAVE xyz, energy, forces, charges ---#
mol = ccwf.molecule()
fxyz = open("GEOM-CCSD.bin", "wb")
ffrc = open("FORC-CCSD.bin", "wb")
fxyz.write(struct.pack('i', mol.natom()))				# number of atoms
ffrc.write(struct.pack('i', mol.natom()))				# number of atoms
fxyz.write(struct.pack('d', ccwf.energy())) # energy

grd = ccwf.gradient().to_array(copy=False, dense=True)

for i in xrange(mol.natom()):
	fxyz.write(struct.pack('i',int(mol.Z(i))))
	fxyz.write(struct.pack('d',mol.x(i)))
	fxyz.write(struct.pack('d',mol.y(i)))
	fxyz.write(struct.pack('d',mol.z(i)))
	ffrc.write(struct.pack('d',-grd[i,0]))
	ffrc.write(struct.pack('d',-grd[i,1]))
	ffrc.write(struct.pack('d',-grd[i,2]))

fxyz.close()
ffrc.close()
'''

