#include "dollar.h"
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cstring>


std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

int prop(float* p0, const float* p1, float *vel){ $
	for(int i3=4; i3 < _nz - 4; i3++){
		for(int i2=4; i2 < _ny - 4; i2++) {
			int ii=i2*_nx + 4 + _n12*i3;
			for(int i1=4; i1 < _nx - 4; i1++,ii++) {
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
}

int prop_tiling(float* p0, const float* p1, float *vel){ $
	int stride = (_nx - 8 )/2;
	for (int ii3 = 4; ii3 < _nz - 4; ii3+=stride) {
	  for (int ii2 = 4; ii2 < _ny - 4; ii2+=stride) {
	      for (int ii1=4; ii1 < _nx - 4; ii1+=stride){
		for (int i3 = ii3; i3 < ii3 + stride; i3++) {
	  	  for (int i2 = ii2; i2 < ii2 + stride; i2++) {
	      		int ii = i2*_nx + _n12*i3 + ii1;
	      		for (int i1=ii1; i1 < ii1 + stride; i1++, ii++){
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
}}}	}
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
	int sz[3] = {128, 128, 128};

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
	size_t align = 0;
#define AVX_SIMD_LENGTH 4
   	p0 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p1 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	vel = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p0_copy = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);

	//std::vector<float, AlignedAllocator<float, align>> p0(n + AVX_SIMD_LENGTH,0.);
	//std::vector<float, AlignedAllocator<float, align>> p1(n + AVX_SIMD_LENGTH,0.);
	//std::vector<float, AlignedAllocator<float, align>> vel(n + AVX_SIMD_LENGTH,0.);
	//std::vector<float, AlignedAllocator<float, align>> p0_copy(n + AVX_SIMD_LENGTH,0.);


	//fill in arrays
	std::random_device rd;
	std::default_random_engine e2(rd());
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0.0, 2.0);
	for (int i = 0; i < n + AVX_SIMD_LENGTH; i++){
		p0[i] = dist(gen);
		p0_copy[i] = p0[i];
		p1[i] = dist(e2);
		vel[i] = dist(e2) * 0.1;
	}

	//align pointers
	p0+=4; p1+=4; vel+=4; p0_copy+=4;
	
	prop(p0_copy, p1, vel);
	prop_tiling(p0, p1, vel);
	check_accuracy(p0_copy, p0, n);

	//for (int i = 0; i < 100; i++)
	//	prop(p0, p1, vel);
	//for (int i = 0; i < 100; i++)
//		prop_tiling(p0, p1, vel);

	//profiling
	dollar::text(std::cout);
	dollar::clear();

	//clean up	
	p0-=4;
	p1-=4;
	vel-=4;
	_mm_free(p0);
	_mm_free(p1);
	_mm_free(vel);
	
	return 0;
}
