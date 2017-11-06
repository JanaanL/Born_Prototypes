#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cstring>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#define STRIDE 4
#define ITER 1
#define OFFSET 0



#define AVX_SIMD_LENGTH 16   // 8 floats 8*4 = 32
#define ALIGNMENT 64 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits

#include <immintrin.h>
#include <zmmintrin.h>


std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

void copy_2D(float* p0, float* b1){ 

	printf("Calling the copy_2D function\n");	
	int stride = STRIDE;
	int blk=0;
	int blk_sz = stride * stride;
	    for(int ii2=0; ii2 < _ny ; ii2+=stride) {
		   for (int ii1 = 0; ii1 < _nx; ii1+=stride, blk++){
			int bidx = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii2,std::min(ii2+stride,_ny)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i2 = r.begin(); i2 != r.end(); ++i2){
			    int pidx = (i2 - 0) * _nx  + ii1 - 0; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1; i1 < std::min(ii1+stride+0, _nx); i1++,pidx++,bidx++) {
			      b1[bidx] = p0[pidx]; 
			     // printf("bidx = %d and pidx = %d\n", bidx, pidx);				
			    
} // i1
} // i2
}); // tbb
} // ii1
} // ii2
} // copy_blocked

void copy_3D(float* p0, float* b1){ 
	int stride = STRIDE;
	
	// foreach block b ...
	// copy data.
	int blk=0;
	int blk_sz = stride * stride * stride;
	 for(int ii3=0; ii3 < _nz ; ii3+=stride) {
	    for(int ii2=0; ii2 < _ny ; ii2+=stride) {
		   for (int ii1 = 0; ii1 < _nx; ii1+=stride, blk++){
		//	printf("The block is now %d\n",blk);
			int bidx = blk*blk_sz;
//			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
//			[&] (const tbb::blocked_range<int>&r){
//			for (int i3 = r.begin(); i3 != r.end(); ++i3){
			for (int i3 = ii3; i3 < ii3 + stride; i3++){
			  for(int i2=ii2; i2 < std::min(ii2+stride+0, _ny); i2++) {
			    int pidx = (i3 -0) * _n12 + (i2 - 0) * _nx  + ii1 - 0; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1; i1 < std::min(ii1+stride+0, _nx); i1++,pidx++,bidx++) {
			      b1[bidx] = p0[pidx]; 
			   //   printf("bidx = %d and pidx = %d\n", bidx, pidx);				
} // i1
} // i2
} // i3
//}); // tbb
} // ii1
} // ii2
} // ii3
} // copy_blocked

void copy_2D_padded(float* p0, float* b1){ 
	int stride = STRIDE;
	
	// set padding
	int offset = OFFSET;
	int blk=0;
	int blk_sz = (stride + offset + offset) * (stride + offset + offset);
	    for(int ii2=offset; ii2 < _ny - offset; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
			int bidx = blk*blk_sz;
		//	printf ("Block %d\n",blk);
			tbb::parallel_for(tbb::blocked_range<int>(ii2 - offset,std::min(ii2+stride + offset,_ny)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i2 = r.begin(); i2 != r.end(); ++i2){
			    int pidx = i2 * _nx  + ii1 - offset; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1 - offset; i1 < std::min(ii1+stride+offset, _nx); i1++,pidx++,bidx++) {
			      b1[bidx] = p0[pidx]; 
			 //     printf("bidx = %d and pidx = %d\n", bidx, pidx);				
} // i1
} // i2
}); // tbb
} // ii1
} // ii2
} // copy_blocked


void copy_3D_padded(float* p0, float* b1){ 
	int stride = STRIDE;
	
	// foreach block b ...
	// copy data.
	int offset = OFFSET;
	int blk=0;
	int blk_sz = (stride + 2 * offset) * (stride + 2 * offset) * (stride + 2 * offset);
	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
	    for(int ii2=offset; ii2 < _ny - offset ; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
		//	printf("The block is now %d\n",blk);
			int bidx = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3 - offset,std::min(ii3+stride+offset,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
			  for(int i2=ii2 - offset; i2 < std::min(ii2+stride+offset, _ny); i2++) {
			    int pidx = (i3 -0) * _n12 + (i2 - 0) * _nx  + ii1 - offset; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1-offset; i1 < std::min(ii1+stride+offset, _nx); i1++,pidx++,bidx++) {
			     	b1[bidx] = p0[pidx]; 
			        printf("bidx = %d and pidx = %d\n", bidx, pidx);			
			        printf("p0[%d] = %f\n",pidx, p0[pidx]);	
			      //  printf("Inside copy function: b1[%d] = %f\n",bidx, b1[bidx]);	
} // i1
} // i2
} // i3
}); // tbb
} // ii1
} // ii2
} // ii3
} // copy_blocked

int prop_block2D(float* p0, float* p2, float *vel){ 
//	born_profiler_t profiler;
//	profiler.start();
	
	//create new blocked array
	int bx = (int) ceil( (float)(_nx) / STRIDE);
	int by = (int) ceil( (float)(_ny) / STRIDE);
	
 	int nBlocks = bx * by;	
	printf("The number of block is %d\n",nBlocks);

	int copy_size = nBlocks * (STRIDE) * (STRIDE);
	printf("the size of the new block is %d\n", copy_size);
   	float *p1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), ALIGNMENT);
	p1+=12;

	copy_2D(p2, p1);
	//for (int i = 0; i < _n12; i++)
	//	printf("p1[%d] = %f",i, p1[i]);
	printf("Finished the blocked copy \n");
	sleep(1);
	int blk=0;
	int blk_sz = (STRIDE) * (STRIDE);
	
	int stride = STRIDE;
		for(int ii2=0; ii2 < _ny; ii2+=stride) {
		   for (int ii1 = 0; ii1 < _nx; ii1+=stride, blk++){
	//		printf("Now at block %d\n",blk);
			int ii = blk*blk_sz;
			//tbb::parallel_for(tbb::blocked_range<int>(ii2,std::min(ii2+stride,_ny)),
			//[&] (const tbb::blocked_range<int>&r){
			//for (int i2 = r.begin(); i2 != r.end(); ++i2){
			for (int i2 = ii2; i2 < ii2+stride; i2++){
				int idx = i2*_nx + ii1;
				for(int i1=ii1; i1 < std::min(ii1+stride, _nx); i1++, ii++, idx++) {
				//	printf("idx = %d\n",idx);
				//	printf("ii = %d\n", ii);

					p0[idx] = p1[ii];
					//printf("p0[%d] = %f\n", idx, p0[idx]); 
					//printf("just finished the calculation\n");
	} //i1
	} //i2
	//} //i3
//	}); //tbb
	} //ii1
	} //ii2
	//} //ii3
	//profiler.read();	
	//profiler.stop();
} //prop

int prop_block3D(float* p0, float* p2, float *vel){ 
//	born_profiler_t profiler;
//	profiler.start();
	
	//create new blocked array
	int bx = (int) ceil( (float)(_nx) / STRIDE);
	int by = (int) ceil( (float)(_ny) / STRIDE);
	int bz = (int) ceil( (float)(_nz) / STRIDE);
	
 	int nBlocks = bx * by * bz;	
	printf("The number of block is %d\n",nBlocks);

	int copy_size = nBlocks * (STRIDE) * (STRIDE) * (STRIDE);
	printf("the size of the new block is %d\n", copy_size);
   	float *p1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), ALIGNMENT);
	p1+=12;

	copy_3D(p2, p1);
	//for (int i = 0; i < _n12; i++)
	//	printf("p1[%d] = %f",i, p1[i]);
	printf("Finished the blocked copy \n");
	sleep(1);
	int blk=0;
	int blk_sz = (STRIDE) * (STRIDE) * (STRIDE);
	
	int stride = STRIDE;
	 for(int ii3=0; ii3 < _nz ; ii3+=stride) {
		for(int ii2=0; ii2 < _ny; ii2+=stride) {
		   for (int ii1 = 0; ii1 < _nx; ii1+=stride, blk++){
	//		printf("Now at block %d\n",blk);
			int ii = blk*blk_sz;
	//		tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
	//		[&] (const tbb::blocked_range<int>&r){
	//		for (int i3 = r.begin(); i3 != r.end(); ++i3){
			for (int i3 = ii3; i3 < std::min(ii3+stride,_nz); i3++){
			  for(int i2=ii2; i2 < std::min(ii2+stride, _ny); i2++) {
					int idx = i3 * _n12 + i2*_nx + ii1;
				    for(int i1=ii1; i1 < std::min(ii1+stride, _nx); i1++, ii++, idx++) {
	//				printf("idx = %d\n",idx);
	//				printf("ii = %d\n", ii);

					p0[idx] = vel[idx] * p1[ii];
					//printf("p0[%d] = %f\n", idx, p0[idx]); 
					//printf("just finished the calculation\n");
	} //i1
	} //i2
	} //i3
//	}); //tbb
	} //ii1
	} //ii2
	} //ii3
	//profiler.read();	
	//profiler.stop();
} //prop

void prop_block_2D_padded(float* p0, float* p2, float *vel){ 
	//create new blocked array
	int bx = (int) ceil( (float)(_nx - 2*OFFSET) / STRIDE);
	int by = (int) ceil( (float)(_ny - 2*OFFSET) / STRIDE);
	
 	int nBlocks = bx * by;	
	printf("The number of block is %d\n",nBlocks);

	int copy_size = nBlocks * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET);
	printf("the size of the new block is %d\n", copy_size);
   	float *p1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), ALIGNMENT);
	p1+=12;

	copy_2D_padded(p2, p1);
	//for (int i = 0; i < _n12; i++)
	//	printf("p1[%d] = %f",i, p1[i]);
	printf("Finished the blocked copy \n");
	sleep(1);
	int stride = STRIDE;
	
	// set padding
	int offset = OFFSET;
	int blk=0;
	int blk_sz = (stride + offset + offset) * (stride + offset + offset);
	    for(int ii2=offset; ii2 < _ny - offset; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
			int ii = blk*blk_sz;
	//		printf ("Block %d\n",blk);
		//	tbb::parallel_for(tbb::blocked_range<int>(ii2,std::min(ii2 + stride,_ny)),
		//	[&] (const tbb::blocked_range<int>&r){
		//	for (int i2 = r.begin(); i2 != r.end(); ++i2){
			for (int i2 = ii2; i2 < std::min(ii2+stride,_ny); i2++){
			    int idx = i2 * _nx  + ii1; 
			    for(int i1=ii1 ; i1 < std::min(ii1+stride, _nx); i1++,ii++,idx++) {
					p0[idx] = vel[idx] * p1[ii];
			//		printf("p0[%d] = %f\n", idx, p0[idx]); 
} // i1
} // i2
//}); // tbb
} // ii1
} // ii2
} // copy_blocked


void prop_block_3D_padded(float* p0, float* p2, float *vel){ 
	//create new blocked array
	int bx = (int) ceil( (float)(_nx - 2*OFFSET) / STRIDE);
	int by = (int) ceil( (float)(_ny - 2*OFFSET) / STRIDE);
	int bz = (int) ceil( (float)(_nz - 2*OFFSET) / STRIDE);
	
 	int nBlocks = bx * by * bz;	
	printf("The number of block is %d\n",nBlocks);

	int copy_size = nBlocks * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET);
	printf("the size of the new block is %d\n", copy_size);
   	float *p1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), ALIGNMENT);
	p1+=12;

	copy_3D_padded(p2, p1);
	//for (int i = 0; i < _n12; i++)
	//	printf("p1[%d] = %f",i, p1[i]);
	printf("Finished the blocked copy \n");
	
	int stride = STRIDE;
	sleep(1);	
	// foreach block b ...
	// copy data.
	int offset = OFFSET;
	int blk=0;
	int blk_sz = (stride + 2 * offset) * (stride + 2 * offset) * (stride + 2 * offset);
	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
	    for(int ii2=offset; ii2 < _ny - offset ; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
	//		printf("The block is now %d\n",blk);
			int ii = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
			//for (int i3 = ii3; i3 < std::min(ii3+stride,_nz); ++i3){
			  for(int i2=ii2; i2 < std::min(ii2+stride, _ny); i2++) {
			    int idx = i3 * _n12 + i2 * _nx  + ii1; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1; i1 < std::min(ii1+stride, _nx); i1++,ii++,idx++) {
					p0[idx] = vel[idx] * p1[ii];
			//		printf("p0[%d] = %f\n", idx, p0[idx]); 
} // i1
} // i2
} // i3
}); // tbb
} // ii1
} // ii2
} // ii3
} // copy_blocked



void check_accuracy(float *p0, int n){

	float epsilon = 0.0001;
	for (int i = 0; i < n; i++){
		if (p0[i] - i > epsilon){
			printf("The accuracy test failed!\n");
			printf("p0[%d] = %f \n", i, p0[i]);
		
	//	return;
		}
	}

	printf("Accuracy check passed!\n");
	return;
}


int main(int argc, char* argv[]) {
	int sz[3] = {16, 16, 16};

	if (argc > 3) {
		sz[0] = std::atoi(argv[1]);
		sz[1] = std::atoi(argv[2]);
		sz[2] = std::atoi(argv[3]);
	}

	int n = sz[0]*sz[1]*sz[2];
	_nx = sz[0];
	_ny = sz[1];
	_nz = sz[2];
	_n12 = _nx * _ny;


   	// allocate memory and align 
	float *p0_3D, *p1_3D, *vel_3D, *p0_copy, *p0_2D, *p1_2D, *vel_2D;
	size_t align = ALIGNMENT;
   	p0_2D = (float *)_mm_malloc((_nx*_ny + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p1_2D = (float *)_mm_malloc((_nx*_ny + AVX_SIMD_LENGTH)*sizeof(float), align);
   	vel_2D = (float *)_mm_malloc((_nx*_ny + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p0_3D = (float *)_mm_malloc((_nx*_ny*_nz + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p1_3D = (float *)_mm_malloc((_nx*_ny*_nz + AVX_SIMD_LENGTH)*sizeof(float), align);
   	vel_3D = (float *)_mm_malloc((_nx*_ny*_nz + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p0_copy = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);

	//align pointers
	p0_3D+=12; p1_3D+=12; vel_3D+=12; p0_copy+=12; p0_2D+=12; p1_2D+=12; vel_2D+=12;

	for (int i = 0; i < _nx*_ny*_nz; i++){
		p1_3D[i] = i;
		vel_3D[i] = 1;
		p0_3D[i] = i;
	}	
	
	for (int i = 0; i < _nx*_ny; i++){
		p1_2D[i] = i;
		vel_2D[i] = 1;
		p0_2D[i] = i;
	}	
	//for (int i = 0; i < n + 4; i++){
	//	p0[i] = i;
	//	vel[i] = 1;
	//}	

	int bx_2D = (int) ceil( (float)(_nx - 2 * OFFSET) / STRIDE);
	int by_2D = (int) ceil( (float)(_ny - 2 * OFFSET) / STRIDE);
	
	int bx_3D = (int) ceil( (float)(_nx - 2 * OFFSET) / STRIDE);
	int by_3D = (int) ceil( (float)(_ny - 2 * OFFSET) / STRIDE);
	int bz_3D = (int) ceil( (float)(_nz - 2 * OFFSET) / STRIDE);
	
 	int nBlocks_2D = bx_2D * by_2D;	
 	int nBlocks_3D = bx_3D * by_3D * bz_3D;	
	int copy_size_2D = nBlocks_2D * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET);	
	int copy_size_3D = nBlocks_3D * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET);	
   	float *b1 = (float *)_mm_malloc((copy_size_2D + AVX_SIMD_LENGTH)*sizeof(float), align);
	b1+=12;
	
	copy_2D(p0_2D, b1);	
	//copy_3D(p0, b1);
	//copy_2D_padded(p0, b1);
	//copy_3D_padded(p0_3D, b1);
	for (int i = 0; i < copy_size_3D; i ++ ){
		printf("After copy function: b1[%d] = %f\n",i, b1[i]);
	}

//	printf("Testing the 2D prop function\n");
//	prop_block2D(p0_2D, p1_2D, vel_2D);
//	check_accuracy(p0_2D, _nx*_ny);
	
//	printf("Testing the 3D prop function\n");
//	prop_block3D(p0_3D, p1_3D, vel_3D);
//	check_accuracy(p0_3D, _nx*_ny*_nz);
	
//	printf("Testing the 2D padded prop function\n");
//	prop_block_2D_padded(p0_2D, p1_2D, vel_2D);
//	check_accuracy(p0_2D, _nx*_ny);
	
//	printf("Testing the 3D padded prop function\n");
//	prop_block_3D_padded(p0_3D, p1_3D, vel_3D);
//	check_accuracy(p0_3D, _nx*_ny*_nz);
	//clean up	
	//p0-=12;
	//p1-=12;
	//_mm_free(p0);
	//_mm_free(p1);
	//_mm_free(vel);
	
	return 0;
}


