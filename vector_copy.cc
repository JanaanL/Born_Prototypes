#include "dollar.h"
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cstring>

//#ifdef U512
//#define AVX_SIMD_LENGTH 8   // 4 doubles 4*8 = 32
//#define VECTOR_LENGTH 16 // 
//#define ALIGNMENT 64 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits

//#define vec_float _m512s
//#else 
#define AVX_SIMD_LENGTH 8   // 8 floats 8*4 = 32
//#define VECTOR_LENGTH 8 // 
#define ALIGNMENT 32 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits
//#define vec_float _m256s
//#define vec_load _mm256_load_ps
//#endif

// #if defined SSE3 || defined AVX
#include <immintrin.h>

// #endif

std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

int prop(float* p0, const float* p1, float *vel){ $
	for(int i3=4; i3 < _nz - 4; i3++){
		for(int i2=4; i2 < _ny - 4; i2++) {
			int ii=i2*_nx + 4 + _n12*i3;
			for(int i1=4; i1 < _nx - 4; i1++,ii++) {
				p0[ii] = vel[ii] * (
				//	(
					coeffs[0]*p1[ii]
					+coeffs[1]*(p1[ii-1]+p1[ii+1])
					+coeffs[2]*(p1[ii-2]+p1[ii+2])
					+coeffs[3]*(p1[ii-3]+p1[ii+3])
					+coeffs[4]*(p1[ii-4]+p1[ii+4])
					
					+coeffs[5]*(p1[ii-1*_nx]+p1[ii+1*_nx])
					+coeffs[6]*(p1[ii-2*_nx]+p1[ii+2*_nx])
					+coeffs[7]*(p1[ii-3*_nx]+p1[ii+3*_nx])
					+coeffs[8]*(p1[ii-4*_nx]+p1[ii+4*_nx])
					
					+coeffs[9]*(p1[ii-1*_n12]+p1[ii+1*_n12])
					+coeffs[10]*(p1[ii-2*_n12]+p1[ii+2*_n12])
					+coeffs[11]*(p1[ii-3*_n12]+p1[ii+3*_n12])
					+coeffs[12]*(p1[ii-4*_n12]+p1[ii+4*_n12])
					)
				+p1[ii] + p1[ii] - p0[ii];
				
				//printf("The new value of p0[%d] is : %f\n", ii, p0[ii]);
			}
		}
	}
}
	

int vector_prop(float* p0, const float* p1, float *vel) { $ 

	//printf("Broadcasting coefficients\n");

	__m256 _coeff0 = _mm256_broadcast_ss(&coeffs[0]);
	__m256 _coeff1 = _mm256_broadcast_ss(&coeffs[1]);
	__m256 _coeff2 = _mm256_broadcast_ss(&coeffs[2]);
	__m256 _coeff3 = _mm256_broadcast_ss(&coeffs[3]);
	__m256 _coeff4 = _mm256_broadcast_ss(&coeffs[4]);
	__m256 _coeff5 = _mm256_broadcast_ss(&coeffs[5]);
	__m256 _coeff6 = _mm256_broadcast_ss(&coeffs[6]);
	__m256 _coeff7 = _mm256_broadcast_ss(&coeffs[7]);
	__m256 _coeff8 = _mm256_broadcast_ss(&coeffs[8]);
	__m256 _coeff9 = _mm256_broadcast_ss(&coeffs[9]);
	__m256 _coeff10 = _mm256_broadcast_ss(&coeffs[10]);
	__m256 _coeff11 = _mm256_broadcast_ss(&coeffs[11]);
	__m256 _coeff12 = _mm256_broadcast_ss(&coeffs[12]);
	//printf("Finished broadcasting coefficients\n");

	__m256 _o, _p0, _p1, _vel, _center;
	__m256 _xm1, _xm2, _xm3, _xm4, _xp1, _xp2, _xp3, _xp4;
	__m256 _ym1, _ym2, _ym3, _ym4, _yp1, _yp2, _yp3, _yp4;
	__m256 _zm1, _zm2, _zm3, _zm4, _zp1, _zp2, _zp3, _zp4;
		    
	for (int i3 = 4; i3 < _nz - 4; i3++) {
	    for (int i2 = 4; i2 < _ny - 4; i2++) {
		int idx = i2*_nx + 4 +_n12*i3;
			for (int i1=4; i1 < _nx - 4; i1+=AVX_SIMD_LENGTH){
			//printf("Loading arrays into vectors\n");
		//	printf("Loop index is %d %d %d",i3,i2, i1);	
				_p1 = _mm256_load_ps( p1 + idx);
				_p0 = _mm256_load_ps( p0 + idx);
				_vel = _mm256_load_ps( vel + idx); 
				_center = _p1;		
			//printf("Finished loading arrays into vectors\n");
				//x values
				_xm1 = _mm256_loadu_ps( p1 + idx - 1);
				_xm2 = _mm256_loadu_ps( p1 + idx - 2);
				_xm3 = _mm256_loadu_ps( p1 + idx - 3);
				_xm4 = _mm256_loadu_ps( p1 + idx - 4);
				_xp1 = _mm256_loadu_ps( p1 + idx + 1);
				_xp2 = _mm256_loadu_ps( p1 + idx + 2);
				_xp3 = _mm256_loadu_ps( p1 + idx + 3);
				_xp4 = _mm256_loadu_ps( p1 + idx + 4);
	//		printf(" Finished Loading x values  into vectors\n");

				//y values
				_ym1 = _mm256_loadu_ps( p1 + idx - (1 * _nx));
				_ym2 = _mm256_loadu_ps( p1 + idx - (2 * _nx));
				_ym3 = _mm256_loadu_ps( p1 + idx - (3 * _nx));
				_ym4 = _mm256_loadu_ps( p1 + idx - (4 * _nx));
				_yp1 = _mm256_loadu_ps( p1 + idx + (1 * _nx));
				_yp2 = _mm256_loadu_ps( p1 + idx + (2 * _nx));
				_yp3 = _mm256_loadu_ps( p1 + idx + (3 * _nx));
				_yp4 = _mm256_loadu_ps( p1 + idx + (4 * _nx));
				
	//		printf(" Finished Loading y values  into vectors\n");
				//x values
				_zm1 = _mm256_loadu_ps( p1 + idx - 1 * _n12);
				_zm2 = _mm256_loadu_ps( p1 + idx - 2 * _n12);
				_zm3 = _mm256_loadu_ps( p1 + idx - 3 * _n12);
				_zm4 = _mm256_loadu_ps( p1 + idx - 4 * _n12);
				_zp1 = _mm256_loadu_ps( p1 + idx + 1 * _n12);
				_zp2 = _mm256_loadu_ps( p1 + idx + 2 * _n12);
				_zp3 = _mm256_loadu_ps( p1 + idx + 3 * _n12);
				_zp4 = _mm256_loadu_ps( p1 + idx + 4 * _n12);
	//		printf(" Finished Loading z values  into vectors\n");

				_o = _mm256_setzero_ps();
				_o = _mm256_fmadd_ps(_coeff0, _center, _o);
				//coeffs[CX1] * (p1[ii-1] + p[ii+i])
				_o = _mm256_fmadd_ps(_coeff1, _xm1, _o);
				_o = _mm256_fmadd_ps(_coeff1, _xp1, _o);
				//coeffs[CX2] * (p1[ii-2] + p[ii+2])
				_o = _mm256_fmadd_ps(_coeff2, _xm2, _o);
				_o = _mm256_fmadd_ps(_coeff2, _xp2, _o);
				//coeffs[CX3] * (p1[ii-3] + p[ii+3])
				_o = _mm256_fmadd_ps(_coeff3, _xm3, _o);
				_o = _mm256_fmadd_ps(_coeff3, _xp3, _o);
				//coeffs[CX4] * (p1[ii-4] + p[ii+4])
				_o = _mm256_fmadd_ps(_coeff4, _xm4, _o);
				_o = _mm256_fmadd_ps(_coeff4, _xp4, _o);
				
				//coeffs[CY1] * (p1[ii-_nx] + p[ii+_nx])
				_o = _mm256_fmadd_ps(_coeff5, _ym1, _o);
				_o = _mm256_fmadd_ps(_coeff5, _yp1, _o);
				//coeffs[CY2] * (p1[ii-2*_nx] + p[ii+2*_nx])
				_o = _mm256_fmadd_ps(_coeff6, _ym2, _o);
				_o = _mm256_fmadd_ps(_coeff6, _yp2, _o);
				//coeffs[CY3] * (p1[ii-3*_nx] + p[ii+3*_nx])
				_o = _mm256_fmadd_ps(_coeff7, _ym3, _o);
				_o = _mm256_fmadd_ps(_coeff7, _yp3, _o);
				//coeffs[CY4] * (p1[ii-4*_nx] + p[ii+4*_nx])
				_o = _mm256_fmadd_ps(_coeff8, _ym4, _o);
				_o = _mm256_fmadd_ps(_coeff8, _yp4, _o);
			
				//coeffs[CZ1] * (p1[ii-_n12] + p[ii+_n12])
				_o = _mm256_fmadd_ps(_coeff9, _zm1, _o);
				_o = _mm256_fmadd_ps(_coeff9, _zp1, _o);
				//coeffs[CZ2] * (p1[ii-_2*n12] + p[ii+_2*n12])
				_o = _mm256_fmadd_ps(_coeff10, _zm2, _o);
				_o = _mm256_fmadd_ps(_coeff10, _zp2, _o);
				//coeffs[CZ3] * (p1[ii-3*_n12] + p[ii+3*_n12])
				_o = _mm256_fmadd_ps(_coeff11, _zm3, _o);
				_o = _mm256_fmadd_ps(_coeff11, _zp3, _o);
				//coeffs[CZ4] * (p1[ii-4*_n12] + p[ii+4*_n12])
				_o = _mm256_fmadd_ps(_coeff12, _zm4, _o);
				_o = _mm256_fmadd_ps(_coeff12, _zp4, _o);

				_o = _mm256_mul_ps(_vel, _o);
				_o = _mm256_add_ps(_p1, _o);
				_o = _mm256_add_ps(_p1, _o);
				_o = _mm256_sub_ps(_o, _p0);
//			printf(" Finished calculations of vectors\n");

				_mm256_stream_ps(p0 + idx, _o);
//			printf(" Finished writing into p0\n");
				//printf("the value of p0[%d] is %f\n", idx, p0[idx]);
				idx+=AVX_SIMD_LENGTH;
			}
		}
	}
}

void check_accuracy(float *orig, float *copy, int n){

	float epsilon = 0.1;
	for (int i = 0; i < n; i++){
		if (orig[i] - copy[i] > epsilon){
			printf("Index %d has a difference of %f\n", i, orig[i] - copy[i]);
			printf("The original prop is %f and the vector prop is %f\n", orig[i], copy[i]); 
		
		return;
		}
	}

	//printf("Printing values from prop function\n");
//	for (int k = 0; k < 12; k++)
		//printf("p0[%d]: %f ", k, p0_copy[k]);
	//printf("\n");
//	printf("Printing values from vector function\n");
//	for (int k = 0; k < 12; k++)
//		printf("p0[%d]: %f ", k, p0[k]);
//	printf("\n");

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

  //printf("Sizing coefficient vector\n");
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

   // allocate 
	float *p0, *p1, *vel, *p0_copy;
	size_t align = ALIGNMENT;


	// malloc, new,   intel aligned malloc 
   p0 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   p1 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   vel = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   p0_copy = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);



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

	p0+=4; p1+=4; vel+=4; p0_copy+=4;
	
	prop(p0_copy, p1, vel);
	vector_prop(p0, p1, vel);
	check_accuracy(p0_copy, p0, n);

	for (int i = 0; i < 100; i++)
		prop(p0, p1, vel);
	for (int i = 0; i < 100; i++)
		vector_prop(p0, p1, vel);

	dollar::text(std::cout);
	dollar::clear();
	
	p0-=4;
	p1-=4;
	vel-=4;
	
	_mm_free(p0);
	_mm_free(p1);
	_mm_free(vel);
	
	
	return 0;
}
