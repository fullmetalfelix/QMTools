
__device__ void kernel_automaton_loadpatch(float *buffer, float *src, uint npts, int3 offset, int3 wo) {

	uint3 gridsize;
	gridsize.x = B * gridDim.x;
	gridsize.y = B * gridDim.y;
	gridsize.z = B * gridDim.z;

	int3 t;
	t.x = threadIdx.x + offset.x + B*blockIdx.x;
	t.y = threadIdx.y + offset.y + B*blockIdx.y;
	t.z = threadIdx.z + offset.z + B*blockIdx.z;

	if(t.x < 0) t.x = gridsize.x - 1;
	else if(t.x == gridsize.x) t.x = 0;

	if(t.y < 0) t.y = gridsize.y - 1;
	else if(t.y == gridsize.y) t.y = 0;

	if(t.z < 0) t.z = gridsize.z - 1;
	else if(t.z == gridsize.z) t.z = 0;

	uint ridx = t.x + t.y*gridsize.x + t.z*gridsize.x*gridsize.y;
	uint widx = (threadIdx.x+wo.x) + (threadIdx.y+wo.y)*BF + (threadIdx.z+wo.z)*BF_2;

	#pragma unroll
	for(ushort k=0; k<4; k++)
		buffer[widx + k*BF_3] = src[ridx + k*npts];

}


__global__ void kernel_automaton_evolve_OLD(
	float 	*qube, 	// input multifield qube
	uint 	npts, 	// number of grid points in a field
	uint 	nf, 	// number of fields
	float 	*qout 	// output multifield qube
	) {

	int ridx, widx;
	__shared__ float buffer[FBUF];
	__shared__ float nnouts[B_3];
	/*	
	__shared__ float *nn;
	ridx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	if(ridx == 0) {
		nn = buffer + BF_3*nf;
	}*/

	// load a patch with fat in shared mem
	
	ridx = (threadIdx.x + blockIdx.x*B);
	ridx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	ridx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	widx = (threadIdx.x+1) + (threadIdx.y+1)*BF + (threadIdx.z+1)*BF_2;

	#pragma unroll
	for(ushort k=0; k<nf; k++) // load the main chunk
		buffer[widx + k*BF_3] = qube[ridx + k*npts];
	__syncthreads();


	/* LOADING EXTRAS:

		tx 	ty 	tz 	bx 	by 	bz 	rx 	ry 	rz
		// FACES
		0 	: 	: 	- 	0 	0 	B 	ty	tz
		1 	: 	: 	+ 	0 	0 	0 	ty	tz
		2 	: 	: 	0 	- 	0 	ty	B 	tz
		3 	: 	: 	0 	+ 	0 	ty 	0 	tz
		4 	: 	: 	0 	0 	- 	tz	ty 	B 
		5 	: 	: 	0 	0 	+ 	tz	ty 	0 

		tx 	ty 	tz 	bx 	by 	bz 	rx 	ry 	rz
		// EDGES
		6 	0 	: 	- 	- 	0 	B 	B 	tz
		6 	1 	: 	- 	+ 	0 	B 	0 	tz
		6 	2 	: 	+ 	- 	0 	0 	B 	tz
		6 	3 	: 	+ 	+ 	0 	0	0 	tz

		6 	4 	: 	0 	- 	- 	tz 	B 	B
		6 	5 	: 	0 	+ 	- 	tz 	0 	B
		6 	6 	: 	0 	- 	+ 	tz 	B 	0
		6 	7 	: 	0 	+ 	+ 	tz 	0 	0

		7 	0 	: 	- 	0 	- 	B 	tz	B
		7 	1 	: 	- 	0 	+ 	B 	tz	0
		7 	2 	: 	+ 	0 	- 	0 	tz	B
		7 	3 	: 	+ 	0 	+ 	0	tz	0
	
		tx 	ty 	tz 	bx 	by 	bz 	rx 	ry 	rz
		// CORNERS

		7 	4 	0 	- 	- 	- 	B 	B	B
		7 	4 	1 	- 	- 	+ 	B 	B	0
		7 	4 	2 	- 	+ 	- 	B 	0 	B
		7 	4 	3 	- 	+ 	+ 	B	0	0
		7 	4 	4 	+ 	- 	- 	0	B	B
		7 	4 	5 	+ 	- 	+ 	0	B	0
		7 	4 	6 	+ 	+ 	- 	0	0	B
		7 	4 	7 	+ 	+ 	+ 	0	0	0
	*/

	int3 b, r, w;

	if(threadIdx.x < 6) { // extra faces

		b.x = (threadIdx.x == 1) - (threadIdx.x == 0);
		b.y = (threadIdx.x == 3) - (threadIdx.x == 2);
		b.z = (threadIdx.x == 5) - (threadIdx.x == 4);

		r.x = (b.x<0)*(B-1) + (b.x==0)*(((threadIdx.x & 2)>0)*threadIdx.y + ((threadIdx.x & 4)>0)*threadIdx.z);
		r.y = (b.y<0)*(B-1) + (b.y==0)*((threadIdx.x < 2)*threadIdx.y + (threadIdx.x > 3)*threadIdx.z);
		r.z = (b.z<0)*(B-1) + (b.z==0)*((threadIdx.x < 4)*threadIdx.z);

		w.x = (b.x>0)*Bp + (b.x==0)*(r.x+1);
		w.y = (b.y>0)*Bp + (b.y==0)*(r.y+1);
		w.z = (b.z>0)*Bp + (b.z==0)*(r.z+1);

	} else if(threadIdx.x == 7 && threadIdx.y == 4) { // extra corners

		b.z = (2*(threadIdx.z & 1)-1);
		b.y = (2*((threadIdx.z & 2)>>1)-1);
		b.x = (2*((threadIdx.z & 4)>>2)-1);

		r.x = (b.x<0)*(B-1);
		r.y = (b.y<0)*(B-1);
		r.z = (b.z<0)*(B-1);

		w.x = (b.x>0)*Bp;
		w.y = (b.y>0)*Bp;
		w.z = (b.z>0)*Bp;

	} else { // edges

		b.x = (2*((threadIdx.y & 2)>0)-1)*(threadIdx.y < 4);
		b.y = (threadIdx.x == 6) * (2*(threadIdx.y & 1)-1);
		if(threadIdx.x == 7) //b.z = (0245 -> -1) (1367 > 1)
			b.z = ((threadIdx.y & 1) > 0)? ((threadIdx.y << 1)+1) : ((threadIdx.y >> 1)+4);
		b.z = (threadIdx.y > 3)*((b.z>5)-(b.z<6));

		r.x = (b.x < 0)*(B-1) + (b.x == 0)*threadIdx.z;
		r.y = (b.y < 0)*(B-1) + (b.y == 0)*threadIdx.z;
		r.z = (b.z < 0)*(B-1) + (b.z == 0)*threadIdx.z;

		w.x = (b.x==0)*(threadIdx.z+1) + (b.x>0)*Bp;
		w.y = (b.y==0)*(threadIdx.z+1) + (b.y>0)*Bp;
		w.z = (b.z==0)*(threadIdx.z+1) + (b.z>0)*Bp;
	
	}

	b.x = blockIdx.x + b.x; b.x = b.x - gridDim.x*(int)floorf((float)b.x / gridDim.x);
	b.y = blockIdx.y + b.y; b.y = b.y - gridDim.y*(int)floorf((float)b.y / gridDim.y);
	b.z = blockIdx.z + b.z; b.z = b.z - gridDim.z*(int)floorf((float)b.z / gridDim.z);

	ridx = (r.x + b.x*B);
	ridx+= (r.y + b.y*B) * gridDim.x * B;
	ridx+= (r.z + b.z*B) * gridDim.x * gridDim.y * B_2;

	widx = w.x + w.y*BF + w.z*BF_2;

	//if(ridx >= npts) {
	//	printf("B%3i %3i %3i T%i%i%i rb %3i%3i%3i -- ridx %i\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, b.x, b.y, b.z, ridx);
	//}
	
	// load
	if(threadIdx.x < 7 || (threadIdx.x == 7 && threadIdx.y == 4)) {
		#pragma unroll
		for(ushort k=0; k<nf; k++) // load the main chunk
			buffer[widx + k*BF_3] = qube[ridx + k*npts];
	}
	__syncthreads();


	/* BREAK POINT!

		now we loaded the patch with the extra border
		we need to compute charge/field transfer with some NN
		but there is not enough shared mem to compute one network per thread!
		so we have to break the mapping and convolve one z-layer at a time.

		for z0 = 0, 1, ... B-1
			thread (x,y,:) computes the full operation for slice z0

	*/




	// load the NN parameters


	// loop over neighbours and evaluate Q transfer network
	ridx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	float result = 0;
	float total = 0;

	short dx,dy,dz;

		#pragma unroll
		for(dx=-1; dx<=1; dx++) {

			#pragma unroll
			for(dy=-1; dy<=1; dy++) {

				#pragma unroll
				for(dz=-1; dz<=1; dz++) {

					//if(dz == 0 && dy == 0 && dx == 0) continue;

					float r = fabsf(dx) + fabsf(dy) + fabsf(dz);
					float factor = exp(-r);
					ushort address = (threadIdx.x+1+dx) + (threadIdx.y+1+dy)*BF + (threadIdx.z+1+dz)*BF_2;

					result += buffer[threadIdx.x+1+dx + (threadIdx.y+1+dy)*BF + (threadIdx.z+1+dz)*BF_2] * factor;
					total += factor;
					//factor = 1.0f / sqrtf(factor);
				
					// evaluate the NN(s)
					// compute layer 1
					for(ushort n=0; n<64; n++) { // loop over neurons
						nnouts[ridx*64 + n] = 0;

						// do some computation for neuron n
						for(ushort k=0; k<nf; k++) {
							nnouts[ridx*64 + n] += 0.1f*buffer[address + BF_3*k];
						}
					}




				}
			}
		}
	


	// DEBUG: simple smoothing kernel *****************
	b.x = threadIdx.x+1;
	b.y = threadIdx.y+1;
	b.z = threadIdx.z+1;

	result = buffer[b.x + (b.y)*BF + (b.z)*BF_2]*6;
	result+= buffer[b.x+1 + (b.y)*BF + (b.z)*BF_2];
	result+= buffer[b.x-1 + (b.y)*BF + (b.z)*BF_2];
	result+= buffer[b.x + (b.y+1)*BF + (b.z)*BF_2];
	result+= buffer[b.x + (b.y-1)*BF + (b.z)*BF_2];
	result+= buffer[b.x + (b.y)*BF + (b.z+1)*BF_2];
	result+= buffer[b.x + (b.y)*BF + (b.z-1)*BF_2];
	result /= 12.0f;
	// ************************************************


	widx = (threadIdx.x + blockIdx.x*B);
	widx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	widx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;

	qout[widx] = result;
}

__device__ int3 kernel_thread_offset(int3 offset) {

	int3 r;

	r.x = threadIdx.x + blockIdx.x*B + offset.x;
	r.y = threadIdx.y + blockIdx.y*B + offset.y;
	r.z = threadIdx.z + blockIdx.z*B + offset.z;

	r.x += ((r.x < 0) - (r.x >= gridDim.x*B))*gridDim.x*B;
	r.y += ((r.y < 0) - (r.y >= gridDim.y*B))*gridDim.y*B;
	r.z += ((r.z < 0) - (r.z >= gridDim.z*B))*gridDim.z*B;

	return r;
}


__global__ void 
__launch_bounds__(512, 4)
kernel_automaton_evolve(
	float 	*qube, 	// input multifield qube
	uint 	npts, 	// number of grid points in a field
	uint 	nf, 	// number of fields
	float 	*qout 	// output multifield qube
	) {

	int ridx, widx;
	__shared__ float buffer[FlatBUF_TOT];
	__shared__ float nnouts[44*B_2];

	int3 b, r, w;

	
	// preload the mid slice and upper fat
	ridx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	b.z = ridx / BF_2;
	b.y = (ridx - b.z * BF_2) / BF;
	b.x = ridx - b.z * BF_2 - b.y * BF;

	widx = b.x + b.y*BF + b.z*BF_2;


	r.x = b.x + blockIdx.x*B - 1;
	r.y = b.y + blockIdx.y*B - 1;
	r.z = b.z + blockIdx.z*B - 1;

	//r.x += ((r.x < 0) - (r.x >= gridDim.x*B))*gridDim.x*B;
	//r.y += ((r.y < 0) - (r.y >= gridDim.y*B))*gridDim.y*B;
	//r.z += ((r.z < 0) - (r.z >= gridDim.z*B))*gridDim.z*B;
	if(r.x < 0) r.x += gridDim.x*B;
	if(r.x >= gridDim.x*B) r.x -= gridDim.x*B;

	if(r.y < 0) r.y += gridDim.y*B;
	if(r.y >= gridDim.y*B) r.y -= gridDim.y*B;

	if(r.z < 0) r.z += gridDim.z*B;
	if(r.z >= gridDim.z*B) r.z -= gridDim.z*B;
	
	ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;

	if(b.z < 3) { // only 3 z-layers will do the load
		/*if(ridx >= npts) {
			printf("B%3i %3i %3i T%i%i%i rt%i %i %i rb %3i%3i%3i -- ridx %i\n", 
				blockIdx.x, blockIdx.y, blockIdx.z, 
				threadIdx.x, threadIdx.y, threadIdx.z, 
				r.x, r.y, r.z,
				b.x, b.y, b.z, ridx);
		}*/
		#pragma unroll
		for(ushort k=0; k<nf; k++)
			buffer[widx + k*FlatBUF] = qube[ridx + k*npts];
	}
	__syncthreads();

	// loop over slices
	for(ushort z=0; z<B; z++) {

		// compute
		// loop over NNs of point tx, ty
		// 		threads (tx,ty,:) compute the result for one point

		short dx,dy,dz;
		//ushort myidx = (threadIdx.x+1) + (threadIdx.y+1)*BF + (threadIdx.z+1)*BF_2;
		//ushort netin = (threadIdx.x + threadIdx.y*B)*44;
		
		#pragma unroll
		for(dx=-1; dx<=1; dx++) { // loop over NNs

			#pragma unroll
			for(dy=-1; dy<=1; dy++) {

				#pragma unroll
				for(dz=-1; dz<=1; dz++) {

					volatile float *output = nnouts + (threadIdx.x + threadIdx.y*B)*44;
					volatile float *inputs = buffer;
					volatile int vol = dx + dy*BF + dz*BF_2 - 4*FlatBUF; // additional offset for the non-central pixel

					// load the inputs for the NN from the buffered patch
					// tz 0,1,2,3 load the central pixel
					// tz 4,5,6,7 load the neighbour pixel
					inputs += (threadIdx.x+1);
					inputs += (threadIdx.y+1)*BF;
					inputs += (threadIdx.z+1)*BF_2;
					inputs += threadIdx.z*FlatBUF;
					inputs += (threadIdx.z > 3)*vol;

					output[threadIdx.z] = inputs[0]; // why are we copying these to another shmem location? because they have to be in order!
					__syncthreads();

					// now the first 8 elements at 
					// nnouts[(threadIdx.x + threadIdx.y*B)*44]
					// contain the inputs




					//if(dz == 0 && dy == 0 && dx == 0) continue;

					/*
					//ushort no = (threadIdx.x + threadIdx.y*B)*44 + 8; // address of the outputs
					#pragma unroll
					for(ushort k=0; k<3; k++) nnouts[(threadIdx.x + threadIdx.y*B)*44 + 8+threadIdx.z+k*B] = 0;
					__syncthreads();
					// **************************
					*/

					// compute the first layer - 8 inputs, 24 outputs
					// parameter matrix is 24 rows of 8 columns
					// output points to the beginning of the input vector for each tx,ty,:
					// output points to the beginning of the input vector for each tx,ty,:
					
					#pragma unroll
					for(ushort k=0; k<3; k++) {
						// ushort row = threadIdx.z+k*B;
						// neuron being updated = threadIdx.z+k*B = row index
						//inputs = output + 20 + k*B + threadIdx.z;
						//inputs[0] = 0;
						float neuronOut = 0;
						volatile ushort os = threadIdx.z+k*B*8;

						#pragma unroll
						for(ushort in=0; in<8; in++) { // loop over the inputs

							neuronOut += c_NN[os+in] * output[in];
							//nnouts[(threadIdx.x + threadIdx.y*B)*44 + 8+threadIdx.z+k*B] += c_NN[threadIdx.z+k*B*8 + in] * nnouts[(threadIdx.x + threadIdx.y*B)*44+in];
						}
						neuronOut += c_NN[8*24+threadIdx.z+k*B]; // add the bias
						neuronOut = tanhf(neuronOut); // 
						os = 20 + k*B + threadIdx.z;
						output[os] = neuronOut;

						//output[20 + k*B + threadIdx.z] = neuronOut;
						//nnouts[(threadIdx.x + threadIdx.y*B)*44 + 8+threadIdx.z+k*B] += c_NN[8*24+threadIdx.z+k*B];
						//nnouts[(threadIdx.x + threadIdx.y*B)*44 + 8+threadIdx.z+k*B] = tanhf(nnouts[(threadIdx.x + threadIdx.y*B)*44 + 8+threadIdx.z+k*B]);
					}
					__syncthreads();

					// now we have all the 24 outputs



					/*

					float r = fabsf(dx) + fabsf(dy) + fabsf(dz);
					float factor = exp(-r);
					ushort address = (threadIdx.x+1+dx) + (threadIdx.y+1+dy)*BF + (threadIdx.z+1+dz)*BF_2;

					result += buffer[threadIdx.x+1+dx + (threadIdx.y+1+dy)*BF + (threadIdx.z+1+dz)*BF_2] * factor;
					total += factor;
					//factor = 1.0f / sqrtf(factor);
				
					// evaluate the NN(s)
					// compute layer 1
					for(ushort n=0; n<64; n++) { // loop over neurons
						nnouts[ridx*64 + n] = 0;

						// do some computation for neuron n
						for(ushort k=0; k<nf; k++) {
							nnouts[ridx*64 + n] += 0.1f*buffer[address + BF_3*k];
						}
					}
					*/
				}
			}
		}


		/*
			// test that the gauss convolution is correct
			if(threadIdx.z == 1) {

				ushort sidx;
				sidx = (threadIdx.x+1) + (threadIdx.y+1)*BF + BF_2;

				
				float result = 0;
				result = buffer[sidx] * 6 + 
					buffer[sidx + 1] + buffer[sidx - 1] +
					buffer[sidx + BF] + buffer[sidx - BF] + 
					buffer[sidx + BF_2] + buffer[sidx - BF_2];

				result /= 12.0f;
				
				int oidx;
				oidx = (threadIdx.x + blockIdx.x*B);
				oidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
				oidx+= (		  z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;

				qout[oidx] = result;
			}
		*/
		__syncthreads();


		// reload the entire patch to see if i was too stupid to code it the smart way!
		// YES! this works so I was really too dumb!
		/*if(b.z < 3) { // only 3 z-layers will do the load
			r.z++;
			if(r.z >= gridDim.z*B) r.z -= gridDim.z*B;
			ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;

			#pragma unroll
			for(ushort k=0; k<nf; k++)
				buffer[widx + k*FlatBUF] = qube[ridx + k*npts];
		}*/
		



		// shift the slices down (towards b.z=0) and load a new 10x10 slice
		// dont do it on the last iteration since there will be no more computing to do
		if(b.z == 0 && z < B-1) {

			int nz = r.z + z + 3;
			if(nz >= gridDim.z*B) nz -= gridDim.z*B;
			nz = r.x + r.y*gridDim.x*B + nz*gridDim.x*gridDim.y*B_2;

			#pragma unroll
			for(ushort k=0; k<nf; k++) {
				buffer[widx + k*FlatBUF] 			= buffer[widx + k*FlatBUF + BF_2];
				buffer[widx + k*FlatBUF + BF_2]		= buffer[widx + k*FlatBUF + 2*BF_2];
				buffer[widx + k*FlatBUF + 2*BF_2] 	= qube[nz + k*FlatBUF];
			}

		}
		__syncthreads();
	}


	/*
		for(b.x=0; b.x<2; b.x++) {
			for(b.y=0; b.y<2; b.y++) {
				for(b.z=0; b.z<2; b.z++) {

					r.x = threadIdx.x - 1 + blockIdx.x*B + b.x*BR;
					r.y = threadIdx.y - 1 + blockIdx.y*B + b.y*BR;
					r.z = threadIdx.z - 1 + blockIdx.z*B + b.z*BR;

					r.x += ((r.x < 0) - (r.x >= gridDim.x*B))*gridDim.x*B;
					r.y += ((r.y < 0) - (r.y >= gridDim.y*B))*gridDim.y*B;
					r.z += ((r.z < 0) - (r.z >= gridDim.z*B))*gridDim.z*B;

					widx = threadIdx.x + threadIdx.y*BR + threadIdx.z*BR_2;
					ridx = r.x + r.y*gridDim.x*B + r.z*gridDim.x*gridDim.y*B_2;

					if(threadIdx.x < 6 && threadIdx.y < 6 && threadIdx.z < 6) {
						#pragma unroll
						for(ushort k=0; k<nf; k++)
							buffer[widx + k*BR_3] = qube[ridx + k*npts];
					}
					__syncthreads();

					// now we have the patch



				}
			}
		}
	*/


	/* BREAK POINT!

		now we loaded the patch with the extra border
		we need to compute charge/field transfer with some NN
		but there is not enough shared mem to compute one network per thread!
		so we have to break the mapping and convolve one z-layer at a time.

		for z0 = 0, 1, ... B-1
			thread (x,y,:) computes the full operation for slice z0

	*/




	// load the NN parameters

	/*
	// loop over neighbours and evaluate Q transfer network
	ridx = threadIdx.x + threadIdx.y*B + threadIdx.z*B_2;
	float result = 0;
	float total = 0;


	


	// DEBUG: simple smoothing kernel *****************
	b.x = threadIdx.x+1;
	b.y = threadIdx.y+1;
	b.z = threadIdx.z+1;

	result = buffer[b.x + (b.y)*BF + (b.z)*BF_2]*6;
	result+= buffer[b.x+1 + (b.y)*BF + (b.z)*BF_2];
	result+= buffer[b.x-1 + (b.y)*BF + (b.z)*BF_2];
	result+= buffer[b.x + (b.y+1)*BF + (b.z)*BF_2];
	result+= buffer[b.x + (b.y-1)*BF + (b.z)*BF_2];
	result+= buffer[b.x + (b.y)*BF + (b.z+1)*BF_2];
	result+= buffer[b.x + (b.y)*BF + (b.z-1)*BF_2];
	result /= 12.0f;
	// ************************************************


	widx = (threadIdx.x + blockIdx.x*B);
	widx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	widx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;

	qout[widx] = result;
	*/
}
