#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cstring>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#define STRIDE 2
#define ITER 1
#define OFFSET 1



#define AVX_SIMD_LENGTH 16   // 8 floats 8*4 = 32
#define ALIGNMENT 64 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits

#include <immintrin.h>
#include <zmmintrin.h>


std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

void copy_2D(float* p0, float* b1){ 
	int stride = STRIDE;
	
	// foreach block b ...
	// copy data.
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
			      printf("bidx = %d and pidx = %d\n", bidx, pidx);				
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
			printf("The block is now %d\n",blk);
			int bidx = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
			  for(int i2=ii2; i2 < std::min(ii2+stride+0, _ny); i2++) {
			    int pidx = (i3 -0) * _n12 + (i2 - 0) * _nx  + ii1 - 0; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1; i1 < std::min(ii1+stride+0, _nx); i1++,pidx++,bidx++) {
			      b1[bidx] = p0[pidx]; 
			      printf("bidx = %d and pidx = %d\n", bidx, pidx);				
} // i1
} // i2
} // i3
}); // tbb
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
			printf ("Block %d\n",blk);
			tbb::parallel_for(tbb::blocked_range<int>(ii2 - offset,std::min(ii2+stride + offset,_ny)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i2 = r.begin(); i2 != r.end(); ++i2){
			    int pidx = i2 * _nx  + ii1 - offset; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1 - offset; i1 < std::min(ii1+stride+offset, _nx); i1++,pidx++,bidx++) {
			      b1[bidx] = p0[pidx]; 
			      printf("bidx = %d and pidx = %d\n", bidx, pidx);				
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
			printf("The block is now %d\n",blk);
			int bidx = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3 - offset,std::min(ii3+stride+offset,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
			  for(int i2=ii2 - offset; i2 < std::min(ii2+stride+offset, _ny); i2++) {
			    int pidx = (i3 -0) * _n12 + (i2 - 0) * _nx  + ii1 - offset; //  + i2 * _nx + i2 * _nx; 
			    for(int i1=ii1-offset; i1 < std::min(ii1+stride+offset, _nx); i1++,pidx++,bidx++) {
			      b1[bidx] = p0[pidx]; 
			    //  printf("bidx = %d and pidx = %d\n", bidx, pidx);			
			    //  printf("p0[%d] = %f\n",pidx, p0[pidx]);	
			      printf("Inside copy function: b1[%d] = %f\n",bidx, b1[bidx]);	
} // i1
} // i2
} // i3
}); // tbb
} // ii1
} // ii2
} // ii3
} // copy_blocked


void copy(float* p0, float* p1){ 
	int stride = STRIDE;
	tbb::parallel_for(tbb::blocked_range<int>(0, _nz),
	[&] (const tbb::blocked_range<int>&r){
	for (int i3 = r.begin(); i3 <r.end(); ++i3){
	
	//for(int i3=4; i3 < _nz - 4; i3++){
		for(int i2=0; i2 < _ny; i2++) {
			int ii=i2*_nx + _n12*i3;
			for(int i1=4; i1 < _nx - 4; i1++,ii++) {
			p0[ii] = p1[i2*stride + stride*stride*i3 + i1];
	}}}}
	);
}




void check_accuracy(float *orig, float *copy, int n){

	float epsilon = 0.0001;
	for (int i = 0; i < n; i++){
		if (orig[i] - copy[i] > epsilon){
			printf("The accuracy test failed!\n");
			printf("Index %d has a difference of %f\n", i, orig[i] - copy[i]);
			printf("The original prop value is %f and the vector prop value is %f\n", orig[i], copy[i]); 
		
		return;
		}
	}

	printf("Accuracy check passed!\n");
	return;
}


int main(int argc, char* argv[]) {
	int sz[3] = {4, 4, 4};

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
	float *p0, *p1, *vel, *p0_copy;
	size_t align = ALIGNMENT;
   	p0 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p1 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p0_copy = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);

	//align pointers
	p0+=12; p1+=12; p0_copy+=12;

	//fill in arrays with index numbers (for debuggin)
	//tbb::parallel_for(tbb::blocked_range<int>(0, n + 4),
	//[&] (const tbb::blocked_range<int>&r){
	//or (int i = r.begin(); i <r.end(); ++i){
	for (int i = 0; i < n + 4; i++){
		p0[i] = i;
//		printf("The value of p0[%d] is %f\n", i, p0[i]);
	}	
	//}); //tbb

	int bx = (int) ceil( (float)(_nx - 2 * OFFSET) / STRIDE);
	int by = (int) ceil( (float)(_ny - 2 * OFFSET) / STRIDE);
	int bz = (int) ceil( (float)(_nz - 2 * OFFSET) / STRIDE);
	
 	int nBlocks = bx * by * bz;	
	int copy_size = nBlocks * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET);	
   	float *b1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), align);
	b1+=12;
	
	//copy_2D(p0, b1);	
	//copy_3D(p0, b1);
	//copy_2D_padded(p0, b1);
	copy_3D_padded(p0, b1);
	for (int i = 0; i < copy_size; i ++ ){
		printf("After copy function: b1[%d] = %f\n",i, b1[i]);
	}

	//clean up	
	p0-=12;
	p1-=12;
	_mm_free(p0);
	_mm_free(p1);
	_mm_free(vel);
	
	return 0;
}


