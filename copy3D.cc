#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cstring>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tick_count.h>
#include "dollar.h"
#define STRIDE 128
#define ITER 1
#define OFFSET 4
#define AVX_SIMD_LENGTH 16
#define ALIGNMENT 64


std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

void copy_3D(const float* p0, float* b1){ $
	
	tbb::tick_count t0 = tbb::tick_count::now();
	int stride = STRIDE;
	int offset = OFFSET;
	
	int blk=0;
	int blk_sz = (stride + 2*offset) * (stride + 2*offset)  * (stride + 2*offset);
	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
	    for(int ii2=offset; ii2 < _ny - offset ; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
			tbb::parallel_for(tbb::blocked_range<int>(ii3 - offset,std::min(ii3+stride + offset,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			int blk_ct = blk*blk_sz;
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
//			for (int i3 = ii3 - offset; i3 < std::min(ii3 + offset + stride,_nz); i3++){
			    int z_ct = (i3 - ii3 + offset);
		            int y_ct = 0;
			    for(int i2=ii2 - offset; i2 < std::min(ii2+stride+offset, _ny); i2++, y_ct++) {
			    	int bidx = blk_ct + y_ct* (stride + 2*offset) + z_ct * (stride + 2*offset)*(stride + 2*offset);
				int pidx = (i3) * _n12 + (i2) * _nx  + ii1 - offset; 
			    	for(int i1=ii1 - offset; i1 < std::min(ii1+stride+offset, _nx); i1++) {
					b1[bidx] = p0[pidx]; 
			//      	printf("p0[%d] = %f\n",pidx, p0[pidx]);	
	//		      		printf("bidx = %d and pidx = %d\n", bidx, pidx);	
					pidx++;
					bidx++;			
} // i1
} // i2
} // i3
}); // tbb
} // ii1
} // ii2
} // ii3

	tbb::tick_count t1 = tbb::tick_count::now();
	std::cout << "Time spent in copy function in seconds: " << (t1-t0).seconds() << std::endl;
} // copy_blocked




int prop_block3D(float* p0, float* p2, float *vel){ $ 

	tbb::tick_count t0 = tbb::tick_count::now();	
	int stride = STRIDE;
	int offset = OFFSET;
	
	//create new blocked array
	int bx = (int) ceil( (float)(_nx - 2 * offset) / stride);
	int by = (int) ceil( (float)(_ny - 2 * offset) / stride);
	int bz = (int) ceil( (float)(_nz - 2 * offset) / stride);
	
 	int nBlocks = bx * by * bz;	
	int blk_sz = (stride + 2*offset) * (stride + 2*offset) * (stride + 2*offset);

	int copy_size = nBlocks * blk_sz;
	printf("the size of the new block is %d\n", copy_size);
   	float *p1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), ALIGNMENT);
	p1+=12;

	copy_3D(p2, p1);
	//for (int i = 0; i < copy_size; i++)
	//	printf("After copy function the value of p1[%d] = %f\n",i, p1[i]);
	printf("Finished the blocked copy \n");
	int blk=0;

	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
		for(int ii2=offset; ii2 < _ny - offset; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
	//		printf("Now at block %d\n",blk);
			int blk_ct = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
	//		for (int i3 = ii3; i3 < std::min(ii3+stride,_nz - offset); i3++){
			     int z_ct = i3 - ii3 + offset;
			     int y_ct = offset;
				for(int i2=ii2; i2 < std::min(ii2+stride, _ny - offset); i2++, y_ct++) {
					int idx = i3 * _n12 + i2*_nx + ii1;
			    		int ii  = blk_ct + y_ct * (stride + 2*offset) + z_ct * (stride + 2*offset)*(stride + 2*offset) + offset;
				    for(int i1=ii1; i1 < std::min(ii1+stride, _nx - offset); i1++, ii++, idx++) {
	//				printf("idx = %d\n",idx);
	//				printf("ii = %d\n", ii);
					//printf("p1[%d] = %f\n",ii,p1[ii]);
					p0[idx] = vel[idx] * p1[ii];
					//printf("p0[%d] = %f\n", idx, p0[idx]); 
	} //i1
	} //i2
	} //i3
	}); //tbb
	} //ii1
	} //ii2
	} //ii3

	tbb::tick_count t1 = tbb::tick_count::now();
	std::cout << "Time spent in prop function in seconds: " << (t1-t0).seconds() << std::endl;
	
} //prop



void check_accuracy(float *p0, int n){
	
	float epsilon = 0.0001;
	for (int i = 0; i < n; i++){
	    if ((int) p0[i] != i && p0[i] != 0.0){
		printf("The accuracy test failed!\n");
		printf("p0[%d] = %f \n", i, p0[i]);
		
		return;
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
	float *p0, *p1, *vel;
	size_t align = ALIGNMENT;
   	p0 = (float *)_mm_malloc((_nx*_ny*_nz + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p1 = (float *)_mm_malloc((_nx*_ny*_nz + AVX_SIMD_LENGTH)*sizeof(float), align);
   	vel = (float *)_mm_malloc((_nx*_ny*_nz + AVX_SIMD_LENGTH)*sizeof(float), align);

	//align pointers
	p0+=12; p1+=12; vel+=12;

	for (int i = 0; i < _nx*_ny*_nz; i++){
		p1[i] = i;
		vel[i] = 1;
		p0[i] = 0;
	 }	
	

	int bx = (int) ceil( (float)(_nx - 2 * OFFSET) / STRIDE);
	int by = (int) ceil( (float)(_ny - 2 * OFFSET) / STRIDE);
	int bz = (int) ceil( (float)(_nz - 2 * OFFSET) / STRIDE);
	
 	int nBlocks = bx * by * bz;	
	int copy_size = nBlocks * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET) * (STRIDE + 2*OFFSET);	
   	float *b1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), align);
	b1+=12;
	
	copy_3D(p1, b1);
	
	//Visually test the results
	//for (int i = 0; i < copy_size; i ++ ){
	//	printf("After copy function: b1[%d] = %f\n",i, b1[i]);
	//}


	printf("Testing the 3D prop function\n");
	prop_block3D(p0, p1, vel);
	check_accuracy(p0, _nx*_ny*_nz);
	
	dollar::text(std::cout);
	dollar::clear();
	//clean up	
	//p0-=12;
	//p1-=12;
	//_mm_free(p0);
	//_mm_free(p1);
	//_mm_free(vel);
	
	return 0;
}


