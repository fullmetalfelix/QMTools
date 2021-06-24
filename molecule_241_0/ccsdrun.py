import psi4
import numpy

psi4.core.set_output_file('ccsdrun.out', False)
psi4.set_memory('24 GB')

mol = psi4.geometry("""
units ang
symmetry c1
#FLAGXYZ
6	-1.2164189302549206 -0.6903094796077225 2.1022886279017772e-06
6	-1.206082694648068 0.7083060988584281 3.3892314461151956e-05
6	-0.010361181192162788 -1.3985964455773356 -6.6001174231387354e-06
6	0.010366976936298887 1.3985959148089768 -1.6191718154627856e-06
6	1.2060847105218244 -0.7083070449688356 -2.34511482930163e-06
6	1.21642180540488 0.6903089731376142 1.1316221631842329e-05
1	-2.1667758140117956 -1.229650438925348 2.517834933085517e-07
1	-2.1483336755499747 1.2616818298526677 4.779536647009002e-05
1	-0.018486579199309367 -2.4912962386910777 -1.3256990626879526e-05
1	0.018473875983203703 2.491296450949431 6.1229239152743e-06
1	2.1483352597699827 -1.2616838655084093 -1.006027774546057e-05
1	2.1667762462400417 1.2296542456716124 1.6905066944670075e-05
""")

mol.fix_com(True)
mol.fix_orientation(True)


psi4.core.set_num_threads(20)
psi4.set_options({'basis': 'cc-pvdz'})
psi4.set_options({'maxiter': 500})
psi4.set_options({'cachelevel': 2})
psi4.set_options({'freeze_core': 'true'})
psi4.set_options({'reference': 'rhf'})

# --- HF calculation --- #
E, wf = psi4.energy('scf', molecule=mol, return_wfn=True)

# save the some matrixes
numpy.save("D-HF", wf.Da().to_array(False, True))
numpy.save("F-HF", wf.Fa().to_array(False, True))
numpy.save("C-HF", wf.Ca().to_array(False, True))
numpy.save("H-HF", wf.H().to_array(False, True))
numpy.save("e-HF", wf.epsilon_a().to_array(False, True))


props, ccwf = psi4.properties('CCSD', molecule=mol, properties=['MULLIKEN_CHARGES'], return_wfn=True, ref_wfn=wf)

# save the some matrixes
numpy.save("D-CCSD", ccwf.Da().to_array(False, True))


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

