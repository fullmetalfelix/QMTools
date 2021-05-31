import psi4
import numpy

psi4.core.set_output_file('ccsdrun.out', False)
psi4.set_memory('24 GB')

mol = psi4.geometry("""
units ang
symmetry c1
#FLAGXYZ
35	2.5775248495833942 0.007162049772548163 -0.009535952472364858
17	-2.2789796151928603 -2.7196725478235626 -0.04642835108172875
17	-2.291413768450889 2.7121379425932863 0.057395817153913994
6	0.66595084380526 0.0029333336998770055 -0.004724364877511964
6	-0.013612880309916671 1.2224836166418627 0.020706949080200144
6	-0.00818740069705264 -1.2196692512609755 -0.025642080158969204
6	-1.4049375273266949 -1.198903355551935 -0.020703940171673634
6	-1.4100984498637859 1.1950734543087818 0.02486291739966955
6	-2.1257977998734843 -0.003438541650202894 0.004294565938281364
1	0.524069001471223 2.168469919563894 0.03721586105891537
1	0.5338258511193738 -2.1632507127289147 -0.04503340385046123
1	-3.2141146782258647 -0.005874623749929786 0.007906311013257434
""")

mol.fix_com(True)
mol.fix_orientation(True)


psi4.core.set_num_threads(24)
psi4.set_options({'basis': 'cc-pvdz'})
psi4.set_options({'maxiter': 500})
psi4.set_options({'cachelevel': 0})
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

