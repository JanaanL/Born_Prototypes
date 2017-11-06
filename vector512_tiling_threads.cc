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
#define STRIDE 112
#define ITER 1
#define OFFSET 0
//#ifdef U512
//#define AVX_SIMD_LENGTH 8   // 4 doubles 4*8 = 32
//#define VECTOR_LENGTH 16 // 
//#define ALIGNMENT 64 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits

//#define vec_float _m512s
//#else 
#define AVX_SIMD_LENGTH 0   // 8 floats 8*4 = 32
//#define VECTOR_LENGTH 8 // 
#define ALIGNMENT 0 // 32 bytes = 256 bits  // avx512 = 64 bytes = 512 bits
//#define vec_float _m256s
//#define vec_load _mm256_load_ps
//#endif

#include <immintrin.h>
#include <zmmintrin.h>

// #endif

std::vector<float> coeffs;
int _nx, _ny, _nz, _n12;

void copy_blocked(const float* p0, float* b1){ $ 
	int stride = STRIDE;
	int offset = 4;
	
	int blk=0;
        int b_nx = (stride + 2*offset);
	int b_n12 = (stride + 2 *offset) * (stride + 2*offset);	
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
			    	int bidx = blk_ct + y_ct*b_nx + z_ct * b_n12;
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
} // copy_blocked

void prop_naive(float *p0, const float *p1, const float *vel){ $
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

int prop_blocked(float* p0, const float* p1, const float *vel){ $ 
	born_profiler_t profiler;
	profiler.start();

	int stride = STRIDE;
	int offset = 4;
	
	//calucalate dimension of new block size
	int blk_sz = (stride + 2*offset) * (stride + 2*offset) * (stride + 2*offset);
	int blk=0;
	int b_nx = stride + 2 * offset;
	int b_n12 = (stride + 2 * offset) * (stride + 2 * offset);

	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
		for(int ii2=offset; ii2 < _ny - offset; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
			int blk_ct = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
		//	for (int i3 = ii3; i3 < std::min(ii3+stride,_nz - offset); i3++){
			     int z_ct = i3 - ii3 + offset;
			     int y_ct = offset;
				for(int i2=ii2; i2 < std::min(ii2+stride, _ny - offset); i2++, y_ct++) {
				    int idx = i3 * _n12 + i2*_nx + ii1;
			    	    int ii  = blk_ct + y_ct * b_nx + z_ct * b_n12 + offset;
				    for(int i1=ii1; i1 < std::min(ii1+stride, _nx - offset); i1++, ii++, idx++) {
					
					p0[idx]=vel[idx]*
					      (
					coeffs[0]*p1[ii]
					+coeffs[1]*(p1[ii-1]+p1[ii+1])+
					+coeffs[2]*(p1[ii-2]+p1[ii+2])+
					+coeffs[3]*(p1[ii-3]+p1[ii+3])+
					+coeffs[4]*(p1[ii-4]+p1[ii+4])+

					+coeffs[5]*(p1[ii-b_nx]+p1[ii+b_nx])+
					+coeffs[6]*(p1[ii-2*b_nx]+p1[ii+2*b_nx])+
					+coeffs[7]*(p1[ii-3*b_nx]+p1[ii+3*b_nx])+
					+coeffs[8]*(p1[ii-4*b_nx]+p1[ii+4*b_nx])+
					
					+coeffs[9]*(p1[ii-1*b_n12]+p1[ii+1*b_n12])+
					+coeffs[10]*(p1[ii-2*b_n12]+p1[ii+2*b_n12])+
					+coeffs[11]*(p1[ii-3*b_n12]+p1[ii+3*b_n12])+
					+coeffs[12]*(p1[ii-4*b_n12]+p1[ii+4*b_n12])
				        )
				        +p1[ii]+p1[ii]-p0[idx];
					//printf("p0[%d] = %f\n", idx, p0[idx]); 
	} //i1
	} //i2
	} //i3
	}); //tbb
	} //ii1
	} //ii2
	} //ii3
	printf("The cache date for the prop_blocked function is \n");
	profiler.read();
	profiler.stop();

} //prop_blocked


int prop_vector(float* p0, const float* p1, float *vel) { $ 
	born_profiler_t profiler;
	profiler.start();

	int stride = STRIDE;
	int offset = 4;
	

	int blk_sz = (stride + 2*offset) * (stride + 2*offset) * (stride + 2*offset);
	int blk=0;
	int b_nx = stride + 2 * offset;
	int b_n12 = (stride + 2 * offset) * (stride + 2 * offset);

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
			    	    int ii  = blk_ct + y_ct * b_nx + z_ct * b_n12 + offset;
				    for(int i1=ii1; i1 < std::min(ii1+stride, _nx - offset); i1+=AVX_SIMD_LENGTH) {
				
				//Broadcast coefficients
				__m512 _coeff0 = _mm512_set1_ps(coeffs[0]);
				__m512 _coeff1 = _mm512_set1_ps(coeffs[1]);
				__m512 _coeff2 = _mm512_set1_ps(coeffs[2]);
				__m512 _coeff3 = _mm512_set1_ps(coeffs[3]);
				__m512 _coeff4 = _mm512_set1_ps(coeffs[4]);
				__m512 _coeff5 = _mm512_set1_ps(coeffs[5]);
				__m512 _coeff6 = _mm512_set1_ps(coeffs[6]);
				__m512 _coeff7 = _mm512_set1_ps(coeffs[7]);
				__m512 _coeff8 = _mm512_set1_ps(coeffs[8]);
				__m512 _coeff9 = _mm512_set1_ps(coeffs[9]);
				__m512 _coeff10 = _mm512_set1_ps(coeffs[10]);
				__m512 _coeff11 = _mm512_set1_ps(coeffs[11]);
				__m512 _coeff12 = _mm512_set1_ps(coeffs[12]);

				__m512 _o, _p0, _p1, _vel, _center;
				__m512 _xm1, _xm2, _xm3, _xm4, _xp1, _xp2, _xp3, _xp4;
				__m512 _ym1, _ym2, _ym3, _ym4, _yp1, _yp2, _yp3, _yp4;
				__m512 _zm1, _zm2, _zm3, _zm4, _zp1, _zp2, _zp3, _zp4;


				//load center values
				_p1 = _mm512_load_ps( p1 + idx);
				_p0 = _mm512_load_ps( p0 + idx);
				_vel = _mm512_load_ps( vel + idx); 
				//_center = _p1;		
				
				//x values
				_xm1 = _mm512_loadu_ps( p1 + idx - 1);
				_xm2 = _mm512_loadu_ps( p1 + idx - 2);
				_xm3 = _mm512_loadu_ps( p1 + idx - 3);
				_xm4 = _mm512_loadu_ps( p1 + idx - 4);
				_xp1 = _mm512_loadu_ps( p1 + idx + 1);
				_xp2 = _mm512_loadu_ps( p1 + idx + 2);
				_xp3 = _mm512_loadu_ps( p1 + idx + 3);
				_xp4 = _mm512_loadu_ps( p1 + idx + 4);

				//y values
				_ym1 = _mm512_loadu_ps( p1 + idx - (1 * b_nx));
				_ym2 = _mm512_loadu_ps( p1 + idx - (2 * b_nx));
				_ym3 = _mm512_loadu_ps( p1 + idx - (3 * b_nx));
				_ym4 = _mm512_loadu_ps( p1 + idx - (4 * b_nx));
				_yp1 = _mm512_loadu_ps( p1 + idx + (1 * b_nx));
				_yp2 = _mm512_loadu_ps( p1 + idx + (2 * b_nx));
				_yp3 = _mm512_loadu_ps( p1 + idx + (3 * b_nx));
				_yp4 = _mm512_loadu_ps( p1 + idx + (4 * b_nx));
				
				//z values
				_zm1 = _mm512_loadu_ps( p1 + idx - 1 * b_n12);
				_zm2 = _mm512_loadu_ps( p1 + idx - 2 * b_n12);
				_zm3 = _mm512_loadu_ps( p1 + idx - 3 * b_n12);
				_zm4 = _mm512_loadu_ps( p1 + idx - 4 * b_n12);
				_zp1 = _mm512_loadu_ps( p1 + idx + 1 * b_n12);
				_zp2 = _mm512_loadu_ps( p1 + idx + 2 * b_n12);
				_zp3 = _mm512_loadu_ps( p1 + idx + 3 * b_n12);
				_zp4 = _mm512_loadu_ps( p1 + idx + 4 * b_n12);

				//calculations:
				_o = _mm512_setzero_ps();
				_o = _mm512_fmadd_ps(_coeff0, _p1, _o);
				//coeffs[CX1] * (p1[ii-1] + p[ii+i])
				_o = _mm512_fmadd_ps(_coeff1, _xm1, _o);
				_o = _mm512_fmadd_ps(_coeff1, _xp1, _o);
				//coeffs[CX2] * (p1[ii-2] + p[ii+2])
				_o = _mm512_fmadd_ps(_coeff2, _xm2, _o);
				_o = _mm512_fmadd_ps(_coeff2, _xp2, _o);
				//coeffs[CX3] * (p1[ii-3] + p[ii+3])
				_o = _mm512_fmadd_ps(_coeff3, _xm3, _o);
				_o = _mm512_fmadd_ps(_coeff3, _xp3, _o);
				//coeffs[CX4] * (p1[ii-4] + p[ii+4])
				_o = _mm512_fmadd_ps(_coeff4, _xm4, _o);
				_o = _mm512_fmadd_ps(_coeff4, _xp4, _o);
				
				//coeffs[CY1] * (p1[ii-_nx] + p[ii+_nx])
				_o = _mm512_fmadd_ps(_coeff5, _ym1, _o);
				_o = _mm512_fmadd_ps(_coeff5, _yp1, _o);
				//coeffs[CY2] * (p1[ii-2*_nx] + p[ii+2*_nx])
				_o = _mm512_fmadd_ps(_coeff6, _ym2, _o);
				_o = _mm512_fmadd_ps(_coeff6, _yp2, _o);
				//coeffs[CY3] * (p1[ii-3*_nx] + p[ii+3*_nx])
				_o = _mm512_fmadd_ps(_coeff7, _ym3, _o);
				_o = _mm512_fmadd_ps(_coeff7, _yp3, _o);
				//coeffs[CY4] * (p1[ii-4*_nx] + p[ii+4*_nx])
				_o = _mm512_fmadd_ps(_coeff8, _ym4, _o);
				_o = _mm512_fmadd_ps(_coeff8, _yp4, _o);
			
				//coeffs[CZ1] * (p1[ii-_n12] + p[ii+_n12])
				_o = _mm512_fmadd_ps(_coeff9, _zm1, _o);
				_o = _mm512_fmadd_ps(_coeff9, _zp1, _o);
				//coeffs[CZ2] * (p1[ii-_2*n12] + p[ii+_2*n12])
				_o = _mm512_fmadd_ps(_coeff10, _zm2, _o);
				_o = _mm512_fmadd_ps(_coeff10, _zp2, _o);
				//coeffs[CZ3] * (p1[ii-3*_n12] + p[ii+3*_n12])
				_o = _mm512_fmadd_ps(_coeff11, _zm3, _o);
				_o = _mm512_fmadd_ps(_coeff11, _zp3, _o);
				//coeffs[CZ4] * (p1[ii-4*_n12] + p[ii+4*_n12])
				_o = _mm512_fmadd_ps(_coeff12, _zm4, _o);
				_o = _mm512_fmadd_ps(_coeff12, _zp4, _o);

				_o = _mm512_mul_ps(_vel, _o);
				_o = _mm512_add_ps(_p1, _o);
				_o = _mm512_add_ps(_p1, _o);
				_o = _mm512_sub_ps(_o, _p0);
				//write final value		
				_mm512_stream_ps(p0 + idx, _o);
				idx+=AVX_SIMD_LENGTH;
				ii+=AVX_SIMD_LENGTH;
	} //i1
	} //i2
	} //i3
	}); //tbb
	} // ii1
	} // ii2
	} // ii3
	printf("The cache date for the prop_vector function is \n");
	profiler.read();
	profiler.stop();
	//profiler.stop();
} //prop_vector

void copy(float* p0, float* p1){ $
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

	float epsilon = 1.0;
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
  // 	p1 = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   //	vel = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
   	p0 = (float *)malloc((n)*sizeof(float));
  	p1 = (float *)malloc((n)*sizeof(float));
   	vel = (float *)malloc((n)*sizeof(float));
   //	p0_copy = (float *)_mm_malloc((n + AVX_SIMD_LENGTH)*sizeof(float), align);
	
	//align pointers
	p0+=OFFSET; p1+=OFFSET; vel+=OFFSET; p0_copy+=OFFSET;


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
		p0_copy[i] = p0[i];
		p1[i] = dist(e2);
		vel[i] = dist(e2) * 0.1;
	}
	});

	int offset = 4;
	int stride = STRIDE;
	//create copy	
	int bx = (int) ceil( (float)(_nx - 2 * offset) / stride);
	int by = (int) ceil( (float)(_ny - 2 * offset) / stride);
	int bz = (int) ceil( (float)(_nz - 2 * offset) / stride);
 	int nBlocks = bx * by * bz;	
	int blk_sz = (stride + 2*offset) * (stride + 2*offset) * (stride + 2*offset);
	int copy_size = nBlocks * blk_sz;
   	//float *b1 = (float *)_mm_malloc((copy_size + AVX_SIMD_LENGTH)*sizeof(float), ALIGNMENT);
	//p1+=OFFSET;

//	copy_blocked(p1, b1);
	//for (int i = 0; i < copy_size; i++)
	//	printf("After copy function the value of p1[%d] = %f\n",i, p1[i]);
	
//	prop_naive(p0, p1, vel);
//	prop_blocked(p0_copy, b1, vel);
//	prop_vector(p0_copy, b1, vel);
//	check_accuracy(p0_copy, p0, n);

	for (int i = 0; i < ITER; i++)
		prop_naive(p0, p1, vel);
	//for (int i = 0; i < ITER; i++)
	//	prop_blocked(p0, p1, vel);
//	for (int i = 0; i < ITER; i++)
//		prop_vector(p0, b1, vel);
//	for (int i = 0; i < ITER; i++)
//		copy_blocked(p1, b1);

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


