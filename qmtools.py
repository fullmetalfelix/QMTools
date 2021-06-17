import numpy
from ctypes import *
import sys
import pickle


ANG2BOR = 1.8897259886


lib = CDLL("./lib/libqmtools.so")

class float3(Structure):
	_fields_ = [
		("x", c_float),
		("y", c_float),
		("z", c_float)
	]

	def __init__(self, arr):
		self.x = c_float(arr[0])
		self.y = c_float(arr[1])
		self.z = c_float(arr[2])

	def __str__(self):
		return "[{}, {}, {}]".format(self.x, self.y, self.z)

	def toArray(self):
		return [self.x, self.y, self.z]

float3_p = POINTER(float3)

class dim3(Structure):
	_fields_ = [
		("x", c_uint),
		("y", c_uint),
		("z", c_uint)
	]

	def __init__(self, x,y,z):
		
		self.x = c_uint(x)
		self.y = c_uint(y)
		self.z = c_uint(z)

	def toArray(self):
		return [self.x, self.y, self.z]

	def Equals(self, other):
		return self.x == other.x and self.y == other.y and self.z == other.z

	def __str__(self):
		return "[{}, {}, {}]".format(self.x, self.y, self.z)

dim3_p = POINTER(dim3)


class short4(Structure):
	_fields_ = [
		("x", c_short),
		("y", c_short),
		("z", c_short),
		("w", c_short),
	]

	def __init__(self, x,y,z,w):
		self.x = c_short(x)
		self.y = c_short(y)
		self.z = c_short(z)
		self.w = c_short(w)

	def __str__(self):
		return "[{} {} {} {}]".format(self.x, self.y, self.z, self.w)
short4_p = POINTER(short4)


class BasisSet(Structure):
	_fields_ = [
		("atomOffset", POINTER(c_int)),
		("nshells", POINTER(c_int)),
		("Ls", POINTER(c_int)),
		("shellOffset", POINTER(c_int)),

		("nparams", c_int),

		("alphas", POINTER(c_float)),
		("coeffs", POINTER(c_float)),
		("global_m_values", POINTER(POINTER(c_int))),

		("d_alphas", POINTER(c_float)),
		("d_coeffs", POINTER(c_float))
	]


	def __init__(self, filename):

		lib.basisset_ini(byref(self), c_char_p(filename.encode("utf-8")))

	def __del__(self):
		
		lib.basisset_del(byref(self))

BasisSet_p = POINTER(BasisSet)
lib.basisset_ini.argtypes = [BasisSet_p, c_char_p]
lib.basisset_del.argtypes = [BasisSet_p]


class Molecule(Structure):
	_fields_ = [
		("natoms", c_int),
		("types", POINTER(c_int)),
		("coords", float3_p), # coords in BOHR relative to grid0

		("qtot", c_int),

		("norbs", c_int),
		("dm", POINTER(c_float)), ("d_dm", POINTER(c_float)),
		("ALMOs", short4_p), ("d_ALMOs", short4_p),

		("basisset", BasisSet_p),

		("d_types", POINTER(c_int)),
		("d_coords", float3_p),
	]

	def __init__(self, filexyz, filedm, basisset):

		print("opening molecule (ang):", filexyz);
		self.basisset = pointer(basisset)

		m = numpy.loadtxt(filexyz)
		
		# make coordinates in bohr
		m[:,1:] *= ANG2BOR
		types = numpy.asarray(m[:,0], dtype=numpy.int32)
		coords= numpy.asarray(m[:,1:], dtype=numpy.float32)
		cf = coords.flatten()
		natoms = types.shape[0]

		c = (float3 * natoms)()
		for i in range(natoms):
			c[i] = float3(cf[3*i:3*i+3])

		self.natoms = c_int(natoms)
		self.types  = types.ctypes.data_as(POINTER(c_int))
		self.coords = c
		self.qtot = c_int(numpy.sum(types))



		# load the DM
		print("opening density matrix:", filedm);
		dm = numpy.load(filedm)
		norbs = dm.shape[0]
		self.norbs = c_int(norbs)
		self.dmatrix = numpy.asarray(dm, dtype=numpy.float32)
		self.dm  = self.dmatrix.ctypes.data_as(POINTER(c_float))
		

		# make the basis
		almos = (short4 * norbs)()
		c = 0

		for a in range(natoms):

			Z = types[a]
			atomOS = basisset.atomOffset[Z]
			nsh = basisset.nshells[Z]
			#print(Z, atomOS, nsh)

			for s in range(nsh):

				L = basisset.Ls[atomOS+s];
				
				# loop over m values
				for mi in range(2*L+1):
					
					almos[c].x = c_short(a)
					almos[c].y = c_short(L)
					almos[c].z = c_short(basisset.global_m_values[L][mi])
					almos[c].w = c_short(basisset.shellOffset[atomOS+s])
					c += 1
		self.ALMOs = almos

		lib.molecule_init(byref(self))


	def __del__(self):

		lib.molecule_del(byref(self))


	def Coords(self):

		xyz = numpy.zeros((self.natoms, 3))
		for i in range(self.natoms):
			xyz[i] = self.coords[i].toArray()

		return numpy.asarray(xyz)

Molecule_p = POINTER(Molecule)
lib.molecule_init.argtypes = [Molecule_p]



class Grid(Structure):
	_fields_ = [
		("shape", dim3),
		("GPUblocks", dim3),
		("npts", c_uint),
		("nfields", c_uint),

		("origin", float3),
		("Ax", float3), ("Ay", float3), ("Az", float3),
		("step", c_float),

		("qube", POINTER(c_float)),
		("d_qube", POINTER(c_float)),
	]

	def __init__(self):

		'''
		self.shape = dim3(nx, ny, nz)
		self.npts = c_uint(nx*ny*nz)

		self._qube = numpy.zeros((nx,ny,nz), dtype=numpy.float32)
		self.qube = self._qube.ctypes.data_as(POINTER(c_float))

		# setup the grid on gpu
		lib.qm_grid_ini(byref(self))
		'''

		pass

		

	def __del__(self):

		# free the curand stuff
		lib.qm_grid_del(byref(self))



	### Creates a cartesian grid around a molecule
	def DensityGrid(molecule, step, fat):

		# get the min and max of each coordinate with some fat
		xyz = molecule.Coords()
		crdmax = numpy.amax(xyz, axis=0) + fat*ANG2BOR
		crdmin = numpy.amin(xyz, axis=0) - fat*ANG2BOR

		print("min {} -- max {}".format(crdmin, crdmax))

		grd = crdmax - crdmin

		grd = grd / (step * ANG2BOR)
		grd = grd / 8.0;
		grd = numpy.ceil(grd).astype(numpy.uint32) * 8
		
		grid = Grid()
		grid.origin = float3(crdmin)
		grid.Ax = float3([1,0,0])
		grid.Ay = float3([0,1,0])
		grid.Az = float3([0,0,1])
		grid.step = c_float(step*ANG2BOR)

		npts = grd[0] * grd[1] * grd[2]
		grid.shape = dim3(grd[0], grd[1], grd[2])
		grid.npts = c_uint(npts)
		grid.nfields = c_uint(1)


		grd = grd / 8
		grd = grd.astype(numpy.uint32)
		grid.GPUblocks 	= dim3(grd[0], grd[1], grd[2])

		grid._qube = numpy.zeros(npts, dtype=numpy.float32)
		grid.qube = grid._qube.ctypes.data_as(POINTER(c_float))

		# gpu allocation
		lib.qm_grid_ini(byref(grid))

		return grid

	def DensityGrid_save(self, filename):

		d = {}
		d["origin"] = self.origin
		d["step"] = self.step
		d["GPUblocks"] = self.GPUblocks
		d["qube"] = self._qube
		d["shape"] = self.shape

		pickle.dump(d, open(filename, "wb"))
		

	def DensityGrid_load(molecule, filename):

		d = pickle.load(open(filename, "rb"))

		grid = Grid()
		grid.origin = d["origin"]
		grid.step = d["step"]
		grid.shape = d["shape"]
		grid.GPUblocks = d["GPUblocks"]
		grid._qube = d["qube"]


		grid.Ax = float3([1,0,0])
		grid.Ay = float3([0,1,0])
		grid.Az = float3([0,0,1])

		npts = grid.shape.x * grid.shape.y * grid.shape.z
		grid.npts = c_uint(npts)
		grid.nfields = c_uint(1)

		grid.qube = grid._qube.ctypes.data_as(POINTER(c_float))

		# gpu allocation
		lib.qm_grid_ini(byref(grid))

		# copy to gpu
		lib.qm_grid_toGPU(byref(grid))

		return grid


	def MultiFieldGrid(molecule, step, fat, nfields=1):

		# get the min and max of each coordinate with some fat
		xyz = molecule.Coords()
		crdmax = numpy.amax(xyz, axis=0) + fat*ANG2BOR
		crdmin = numpy.amin(xyz, axis=0) - fat*ANG2BOR

		print("min {} -- max {}".format(crdmin, crdmax))

		grd = crdmax - crdmin

		grd = grd / (step * ANG2BOR)
		grd = grd / 8.0;
		grd = numpy.ceil(grd).astype(numpy.uint32) * 8
		
		grid = Grid()
		grid.origin = float3(crdmin)
		grid.Ax = float3([1,0,0])
		grid.Ay = float3([0,1,0])
		grid.Az = float3([0,0,1])
		grid.step = c_float(step*ANG2BOR)

		npts = grd[0] * grd[1] * grd[2]
		grid.shape = dim3(grd[0], grd[1], grd[2])
		grid.npts = c_uint(npts)
		grid.nfields = c_uint(nfields)

		grd = grd / 8
		grd = grd.astype(numpy.uint32)
		grid.GPUblocks 	= dim3(grd[0], grd[1], grd[2])


		grid._qube = numpy.zeros(npts*nfields, dtype=numpy.float32)
		grid.qube = grid._qube.ctypes.data_as(POINTER(c_float))

		# gpu allocation
		lib.qm_grid_ini(byref(grid))

		return grid





	def TestGrid(molecule, step, fat):

		crdmin = numpy.asarray([-12,-11,-8])
		crdmax = -crdmin

		print("min {} -- max {}".format(crdmin, crdmax))

		grd = crdmax - crdmin

		grd = grd / (step * ANG2BOR)
		grd = grd / 8.0;
		grd = numpy.ceil(grd).astype(numpy.uint32) * 8
		
		grid = Grid()
		grid.origin = float3(crdmin)
		grid.Ax = float3([1,0,0])
		grid.Ay = float3([0,1,0])
		grid.Az = float3([0,0,1])
		grid.step = c_float(step*ANG2BOR)

		npts = grd[0] * grd[1] * grd[2]

		grid.shape = dim3(grd[0], grd[1], grd[2])
		grid.npts = c_uint(npts)

		grd = grd / 8
		grd = grd.astype(numpy.uint32)
		grid.GPUblocks 	= dim3(grd[0], grd[1], grd[2])

		#grid._qube = numpy.zeros(tuple(grid.shape.toArray()), dtype=numpy.float32, order='F') #, dtype=numpy.float32
		#grid.qube = grid._qube.ctypes.data_as(POINTER(c_float))
		grid._qube = numpy.zeros(npts, dtype=numpy.float32)
		grid.qube = grid._qube.ctypes.data_as(POINTER(c_float))

		# gpu allocation
		lib.qm_grid_ini(byref(grid))

		return grid




	# create a grid with custom shape
	def MakeGrid(origin, step, shape):

		x0 = numpy.asarray(origin) * ANG2BOR

		grd = numpy.asarray(shape)
		grd = grd / 8.0;
		grd = numpy.ceil(grd).astype(numpy.uint32) * 8
		
		grid = Grid()
		grid.origin = float3(x0)
		grid.Ax = float3([1,0,0])
		grid.Ay = float3([0,1,0])
		grid.Az = float3([0,0,1])
		grid.step = c_float(step*ANG2BOR)

		npts = grd[0] * grd[1] * grd[2]
		grid.shape = dim3(grd[0], grd[1], grd[2])
		grid.npts = c_uint(npts)

		grd = grd / 8
		grd = grd.astype(numpy.uint32)
		grid.GPUblocks 	= dim3(grd[0], grd[1], grd[2])

		grid._qube = numpy.zeros(npts, dtype=numpy.float32)
		grid.qube = grid._qube.ctypes.data_as(POINTER(c_float))

		# gpu allocation
		lib.qm_grid_ini(byref(grid))

		return grid


	def __str__(self):
		return "origin: {}\nshape: {} -- points: {} gpublocks: {}".format(self.origin, self.shape, self.npts, self.GPUblocks)



Grid_p = POINTER(Grid)
lib.qm_grid_ini.argtypes = [Grid_p]
lib.qm_grid_del.argtypes = [Grid_p]
lib.qm_grid_toGPU.argtypes = [Grid_p]


class QMTools(Structure):
	_fields_ = [
		("nrpts", c_int),
		("nacsf", c_int),

		("qubebuffer", POINTER(c_float)),
		("acsfbuffer", POINTER(c_float)),
		("rptsbuffer", float3_p),

		("curandstate", POINTER(c_int)),
	]

	RPTSperATOM = 128
	NACSF = 81

	def __init__(self):


		# setup the curand stuff
		lib.qm_ini(byref(self))

		# allocate
		self.locations = None

		

	def __del__(self):

		# free the curand stuff
		lib.qm_del(byref(self))

		pass


	def compute(self, molecule):

		natm = molecule.natoms
		nrpts = natm * SCSF.RPTSperATOM
		nacsf = nrpts * SCSF.NACSF

		self.nrpts = c_int(nrpts)
		self.nacsf = c_int(nacsf)

		self.qube = numpy.zeros(nrpts, dtype=numpy.float32)
		self.qubebuffer = self.qube.ctypes.data_as(POINTER(c_float))

		self.acsf = numpy.zeros((nrpts, SCSF.NACSF), dtype=numpy.float32)
		self.acsfbuffer = self.acsf.ctypes.data_as(POINTER(c_float))

		self.rptsbuffer = (float3 * nrpts)()


		lib.scsf_compute(byref(self), byref(molecule))


		#self.locations = numpy.ctypeslib.as_array(self.rptsbuffer, shape=(nrpts,3))
		self.locations = numpy.zeros((nrpts,3), dtype=numpy.float32)
		for i in range(nrpts):
			self.locations[i] = self.rptsbuffer[i].toArray()

		#print(self.acsf)
		#print(self.locations)
		#print(self.acsf[0])
		#print(self.qube, numpy.max(self.qube))


	def ComputeDensity(self, molecule, grid):

		lib.qm_densityqube(byref(molecule), byref(grid))
		return numpy.copy(grid._qube)

	def ComputeDensity_subgrid(self, molecule, grid):

		lib.qm_densityqube_subgrid(byref(molecule), byref(grid))
		return numpy.copy(grid._qube)

	
	def ComputeDensity_shmem(self, molecule, grid):

		lib.qm_densityqube_shmem(byref(molecule), byref(grid))
		return numpy.copy(grid._qube)
	

	def WriteDensity(self, molecule, grid, filename):

		# create byte objects from the strings
		b_string = c_char_p(filename.encode('utf-8'))

		# send strings to c function
		lib.qm_gridmol_write(byref(grid), byref(molecule), b_string)


	def ComputeHartree(self, molecule, densitygrid, grid):

		lib.qm_hartree(byref(molecule), byref(densitygrid), byref(grid))
		return numpy.copy(grid._qube)


	def poisson_fft(self, mol,  grid, omega2=[]):
		grid_vec = numpy.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1]
        ])
		grid_origin=(grid.origin.x, grid.origin.y, grid.origin.z)
		grid_shape=(grid.shape.x,grid.shape.y,grid.shape.z)
		scan_window=(
        	grid_origin,
        	tuple([grid_origin[i] + grid_vec[i][i]*grid_shape[i] for i in range(3)])
    	)
		ndim = 3
		L = [scan_window[1][i]-scan_window[0][i] for i in range(ndim )]
		N = (120, 120, 64)
		omega2 = [-(2*numpy.pi*numpy.fft.fftfreq(N[i], L[i]/N[i])) ** 2 for i in range(ndim)]
		omega2 = numpy.ravel(numpy.stack(numpy.meshgrid(*omega2, indexing='ij'), axis=-1).sum(axis=-1).astype(numpy.float32), order='F')

		lib.poisson_fft(byref(mol), byref(grid), omega2.ctypes.data_as(POINTER(c_float)))




QMTools_p = POINTER(QMTools)
#lib.scsf_compute.argtypes = [QMTools_p, Molecule_p]
lib.qm_ini.argtypes = [QMTools_p]
lib.qm_del.argtypes = [QMTools_p]
lib.qm_densityqube.argtypes = [Molecule_p, Grid_p]
lib.qm_densityqube_subgrid.argtypes = [Molecule_p, Grid_p]
lib.qm_densityqube_shmem.argtypes = [Molecule_p, Grid_p]
lib.qm_hartree.argtypes = [Molecule_p, Grid_p, Grid_p]
lib.qm_gridmol_write.argtypes = [Grid_p, Molecule_p, c_char_p]

lib.poisson_fft.argtypes = [Molecule_p, Grid_p, POINTER(c_float)]
#lib.DQ_sum.argtypes = [Molecule_p, Grid_p]



lib.automaton_set_NN.argtypes = [POINTER(c_float), c_int]
lib.automaton_reset.argtypes = [Grid_p]
lib.automaton_compute_vnn.argtypes = [Grid_p, Molecule_p]
lib.automaton_compute_qseed.argtypes = [Grid_p, Molecule_p]

lib.automaton_compute_evolution.argtypes = [Grid_p, Molecule_p, c_int, c_int]
lib.automaton_compute_evolution.restype = c_float

lib.automaton_compare.argtypes = [Grid_p, Grid_p]
lib.automaton_compare.restype = c_float


class QMAutomaton(object):

	def __init__(self):

		pass


	def SetNetwork(parameters):

		e = numpy.asarray(parameters, dtype=numpy.float32)

		# set the parameters in the library
		lib.automaton_set_NN(e.ctypes.data_as(POINTER(c_float)), c_int(e.shape[0]))

	def ResetQAB(grid, molecule):

		# set all to 0
		lib.automaton_reset(byref(grid))

		# create a q seed
		lib.automaton_compute_qseed(byref(grid), byref(molecule))


	def ComputeVNe(grid, molecule):

		lib.automaton_compute_vnn(byref(grid), byref(molecule))


	def Evolve(grid, molecule, maxiter=1, copyback=0, saveall=False):

		for i in range(maxiter):
			delta = lib.automaton_compute_evolution(byref(grid), byref(molecule), c_int(1), c_int(copyback))
			if saveall == True:
				numpy.save("grid.evo.trained-{0:04d}.npy".format(i), grid._qube)

		return delta

	def Evolve_conv(grid, molecule, maxiter=1000, tolerance=1.0e-16):

		nbatch = 300
		iters = 0
		delta = 1
		while delta > tolerance:
			delta = lib.automaton_compute_evolution(byref(grid), byref(molecule), c_int(nbatch), c_int(0))
			iters += nbatch
			if iters >= maxiter: break

		print(iters, delta < tolerance)
		lib.automaton_compute_evolution(byref(grid), byref(molecule), c_int(1), c_int(1))

		return delta, iters

	def Compare(refgrid, grid):

		if not grid.shape.Equals(refgrid.shape):
			print("compare: grids do not have same shape!",grid.shape, refgrid.shape)
			return None

		delta = lib.automaton_compare(byref(grid), byref(refgrid))
		return delta



	def Mix(e1, e2, mutationRate, mutationSize):

		dnasize = e1.shape[0]
		mask = numpy.random.choice([0,1], dnasize)

		parents = [e1,e2]
		son = numpy.zeros(dnasize)
		for i in range(dnasize): son[i] = parents[mask[i]][i]
		son = numpy.asarray(son, dtype=numpy.float32)

		mp = mutationRate / dnasize
		mutations = (2*numpy.random.rand(dnasize) - 1) * mutationSize
		mask = numpy.random.choice([0,1], dnasize, p=[1-mp, mp])
		mutations *= mask

		son += mutations
		# make sure the last 2 are positive numbers in 0-1
		son[-2:] = numpy.abs(son[-2:])
		if son[-1] > 1: son[-1] -= 1
		if son[-2] > 1: son[-2] -= 1

		return son

	def Rand(dnasize, dnamax=2.5):

		e = numpy.random.rand(dnasize)
		e[0:-2] = (2*e[0:-2]-1)*dnamax # makes sure the last 2 parameters are positive

		return e
