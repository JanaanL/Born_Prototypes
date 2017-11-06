#include "dollar.h"
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cstring>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include "born_profiler.h"
#define ITER 1
#define OFFSET 0

#define AVX_SIMD_LENGTH 16   // 8 floats 8*4 = 32
#define ALIGNMENT 64 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits
#include <immintrin.h>
#include <zmmintrin.h>


std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

int prop(float* p0, float* p1, float *vel){ $
	born_profiler_t profiler;
	profiler.start();
	
	tbb::parallel_for(tbb::blocked_range<int>(4, _nz - 4),
	[&] (const tbb::blocked_range<int>&r){
	for (int i3 = r.begin(); i3 <r.end(); ++i3){
	
	//for(int i3=4; i3 < _nz - 4; i3++){
		for(int i2=4; i2 < _ny - 4; i2++) {
			int ii=i2*_nx + 4 + _n12*i3;
			for(int i1=4; i1 < _nx - 4; i1++,ii++) {
				float x = 
				p0[ii] = vel[ii] * (
					coeffs[0]*p1[ii]
					+coeffs[1]*(p1[ii-1]+p1[ii+1])+
					+coeffs[2]*(p1[ii-2]+p1[ii+2])+
					+coeffs[3]*(p1[ii-3]+p1[ii+3])+
					+coeffs[4]*(p1[ii-4]+p1[ii+4])+
					
					+coeffs[5]*(p1[ii-1*_nx]+p1[ii+1*_nx])+
					+coeffs[6]*(p1[ii-2*_nx]+p1[ii+2*_nx])+
					+coeffs[7]*(p1[ii-3*_nx]+p1[ii+3*_nx])+
					+coeffs[8]*(p1[ii-4*_nx]+p1[ii+4*_nx])+
					
					+coeffs[9]*(p1[ii-1*_n12]+p1[ii+1*_n12])+
					+coeffs[10]*(p1[ii-2*_n12]+p1[ii+2*_n12])+
					+coeffs[11]*(p1[ii-3*_n12]+p1[ii+3*_n12])+
					+coeffs[12]*(p1[ii-4*_n12]+p1[ii+4*_n12])
					)
				+p1[ii] + p1[ii] - p0[ii];
			}
		}
	}
});
	profiler.read();
	profiler.stop();
}


void prop_naive(float *p0, float *p1, float *vel){ $
	born_profiler_t profiler;
	profiler.start();

	tbb::parallel_for(tbb::blocked_range<int>(4,_nz-4),[&](
  	const tbb::blocked_range<int>&r){
  	for(int  i3=r.begin(); i3!=r.end(); ++i3){
//	for(int i3=4; i3 < _nz-4; i3++){
		for(int i2=4; i2 < _ny-4; i2++) {
			int ii=i2*_nx+4+_n12*i3;
			for(int i1=4; i1 < _nx-4; i1++,ii++) {
		        float x=
				p0[ii]=vel[ii]*
					      (
					coeffs[0]*p1[ii]
					+coeffs[1]*(p1[ii-1]+p1[ii+1])+
					+coeffs[2]*(p1[ii-2]+p1[ii+2])+
					+coeffs[3]*(p1[ii-3]+p1[ii+3])+
					+coeffs[4]*(p1[ii-4]+p1[ii+4])+
					+coeffs[5]*(p1[ii-_nx]+p1[ii+_nx])+
					+coeffs[6]*(p1[ii-2*_nx]+p1[ii+2*_nx])+
					+coeffs[7]*(p1[ii-3*_nx]+p1[ii+3*_nx])+
					+coeffs[8]*(p1[ii-4*_nx]+p1[ii+4*_nx])+
					+coeffs[9]*(p1[ii-1*_n12]+p1[ii+1*_n12])+
					+coeffs[10]*(p1[ii-2*_n12]+p1[ii+2*_n12])+
					+coeffs[11]*(p1[ii-3*_n12]+p1[ii+3*_n12])+
					+coeffs[12]*(p1[ii-4*_n12]+p1[ii+4*_n12])
				        )
				        +p1[ii]+p1[ii]-p0[ii];
				      
			}
		}
	}
	});
	printf("The cache date for the prop_naive function is \n");
	profiler.read();
	profiler.stop();
}


int main(int argc, char* argv[]) {
	int sz[3] = {256, 256, 256};

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
  	
	//fill coefficients with values
	coeffs.clear();
  	coeffs.resize(13);
  	coeffs[0] = 1;
  	coeffs[1] = 2;
  	coeffs[2] = 3;
 	coeffs[4] = 4;
  	coeffs[5] = 5;
  	coeffs[6] = 6;
  	coeffs[7] = 7;
  	coeffs[8] = 8;
  	coeffs[9] = 9;
  	coeffs[10] = 10;
  	coeffs[11] = 11;
  	coeffs[12] = 12;

   	// allocate memory and align 
	float *p0, *p1, *vel, *p0_copy;
	size_t align = ALIGNMENT;
//   	p0 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
//   	p1 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
//   	vel = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p0 = (float *)malloc((n)*sizeof(float));
  	p1 = (float *)malloc((n)*sizeof(float));
   	vel = (float *)malloc((n)*sizeof(float));
	
	//align pointers
	p0+=OFFSET; p1+=OFFSET; vel+=OFFSET;


	//fill in arrays
	std::random_device rd;
	std::default_random_engine e2(rd());
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0.0, 2.0);

	tbb::parallel_for(tbb::blocked_range<int>(0,n + AVX_SIMD_LENGTH),
	[&] (const tbb::blocked_range<int>&r){
	for (int i = r.begin(); i != r.end(); ++i){
	//for (int i = 0; i < AVX_SIMD_LENGTH + n; i++){
		p0[i] = dist(gen);
		p1[i] = dist(e2);
		vel[i] = dist(e2) * 0.1;
	}
	});



	for (int i = 0; i < ITER; i++)
		prop_naive(p0, p1, vel);
	for (int i = 0; i < ITER; i++)
		prop(p0, p1, vel);

	//profiling
	dollar::text(std::cout);
	dollar::clear();

	//clean up	
	//p0-=12;
	//p1-=12;
	//vel-=12;
	//_mm_free(p0);
	//_mm_free(p1);
	//_mm_free(vel);
	
	return 0;
}


