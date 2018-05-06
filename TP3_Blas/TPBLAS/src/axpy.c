#include "mnblas.h"
#include "complex.h"
#include <omp.h>
#include <x86intrin.h>

void mncblas_saxpy(const int N, const float a, const float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      Y [j] += a* X[i];
    }

  return ;
}

void mncblas_saxpy_static(const int N, const float a, const float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i ;
#pragma omp parallel for schedule(static)
  for (i=0; i < N ; i += incX)
    {
      Y [i] += a * X[i];
    }

  return ;
}

void mncblas_saxpy_vector(const int N, const float a, const float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i;
	__m128 v1,v2;
	float tab_a[4];
	tab_a[0]=a;tab_a[1]=a;tab_a[2]=a;tab_a[3]=a;
#pragma omp parallel for schedule(static) private(v1,v2)
  for (i=0; i < N ; i += 4*incX)
    {
			v1 = _mm_load_ps(&X[i]);
			v2 = _mm_load_ps(tab_a);
			v1 = _mm_mul_ps(v1,v2);

      Y [i] += v1[0];
			Y [i+1] += v1[1];
			Y [i+2] += v1[2];
			Y [i+3] += v1[3];
    }

  return ;
}

void mncblas_daxpy(const int N, const double a, const double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      Y [j] = a *X[i] + Y[j] ;
    }

  return ;
}

void mncblas_daxpy_static(const int N, const double a, const double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
#pragma omp parallel for schedule(static)
  for (i=0; i < N ; i += 8*incX)
    {
      Y [i] += a* X[i];
			Y [i+1] += a* X[i+1];
			Y [i+2] += a* X[i+2];
			Y [i+3] += a* X[i+3];
			Y [i+4] += a* X[i+4];
			Y [i+5] += a* X[i+5];
			Y [i+6] += a* X[i+6];
			Y [i+7] += a* X[i+7];
    }

  return ;
}

void mncblas_daxpy_vector(const int N, const double a, const double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i ;
	__m128d v1,v2;
	double tab_a[2];
	tab_a[0]=a;tab_a[1]=a;
#pragma omp parallel for schedule(static) private(v1,v2)
  for (i=0; i < N ; i += 2*incX)
    {
			v1 = _mm_load_pd(&X[i]);
			v2 = _mm_load_pd(tab_a);
			v1 = _mm_mul_pd(v1,v2);

      Y [i] += v1[0];
			Y [i+1] += v1[1];
    }

  return ;
}


void mncblas_caxpy(const int N, const void* a, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      ((struct complex_simple*)Y)[j] = addition_cs( ((struct complex_simple*)Y)[j] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i] ) );
    }

  return ;
}

void mncblas_caxpy_vector(const int N, const void* a, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

	
	float *tab_X = (float*)X;
	__m128 v1,v2,v3;
	float tab_a[4];
	tab_a[0]=((struct complex_simple*)a)->real;
	tab_a[1]=((struct complex_simple*)a)->imaginary;
	tab_a[2]=((struct complex_simple*)a)->real;
	tab_a[3]=((struct complex_simple*)a)->imaginary;

	float tab_ai[4];
	tab_ai[0]=((struct complex_simple*)a)->imaginary;
	tab_ai[1]=((struct complex_simple*)a)->real;
	tab_ai[2]=((struct complex_simple*)a)->imaginary;
	tab_ai[3]=((struct complex_simple*)a)->real;

#pragma omp parallel for schedule(static) private(v1,v2,v3)
  for (i=0; i < 2*N ; i += 4*incX)
    {
      
			v1 = _mm_load_ps(&tab_X[i]);

			v2 = _mm_load_ps(tab_a);
			v3 = _mm_mul_ps(v1,v2);

			((float*)Y) [i] += (v3[0] - v3[1]);
			((float*)Y) [i+2] += (v3[2] - v3[3]);

			v2 = _mm_load_ps(tab_ai);
			v3 = _mm_mul_ps(v1,v2);

			((float*)Y) [i+1] += (v3[0] + v3[1]);
			((float*)Y) [i+3] += (v3[2] + v3[3]);

    }

  return ;
}



void mncblas_caxpy_static(const int N, const void* a, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i ;
#pragma omp parallel for schedule(static)
  for (i=0; i < N ; i += 8*incX)
    {
      ((struct complex_simple*)Y)[i] = addition_cs( ((struct complex_simple*)Y)[i] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i] ) );
      ((struct complex_simple*)Y)[i+1] = addition_cs( ((struct complex_simple*)Y)[i+1] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+1] ) );
      ((struct complex_simple*)Y)[i+2] = addition_cs( ((struct complex_simple*)Y)[i+2] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+2] ) );
      ((struct complex_simple*)Y)[i+3] = addition_cs( ((struct complex_simple*)Y)[i+3] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+3] ) );
      ((struct complex_simple*)Y)[i+4] = addition_cs( ((struct complex_simple*)Y)[i+4] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+4] ) );
      ((struct complex_simple*)Y)[i+5] = addition_cs( ((struct complex_simple*)Y)[i+5] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+5] ) );
      ((struct complex_simple*)Y)[i+6] = addition_cs( ((struct complex_simple*)Y)[i+6] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+6] ) );
      ((struct complex_simple*)Y)[i+7] = addition_cs( ((struct complex_simple*)Y)[i+7] , multiplication_cs( *((struct complex_simple*)a) , ((struct complex_simple*)X)[i+7] ) );
    }

  return ;
}

void mncblas_zaxpy(const int N, const void* a, const void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      ((struct complex_double*)Y)[j] = addition_cd( ((struct complex_double*)Y)[j] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i] ) );
    }

  return ;
}

void mncblas_zaxpy_static(const int N, const void* a, const void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
#pragma omp parallel for schedule(static)
  for (i=0; i < N ; i += 8*incX)
    {
      ((struct complex_double*)Y)[i] = addition_cd( ((struct complex_double*)Y)[i] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i] ) );
      ((struct complex_double*)Y)[i+1] = addition_cd( ((struct complex_double*)Y)[i+1] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+1] ) );
      ((struct complex_double*)Y)[i+2] = addition_cd( ((struct complex_double*)Y)[i+2] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+2] ) );
      ((struct complex_double*)Y)[i+3] = addition_cd( ((struct complex_double*)Y)[i+3] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+3] ) );
      ((struct complex_double*)Y)[i+4] = addition_cd( ((struct complex_double*)Y)[i+4] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+4] ) );
      ((struct complex_double*)Y)[i+5] = addition_cd( ((struct complex_double*)Y)[i+5] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+5] ) );
      ((struct complex_double*)Y)[i+6] = addition_cd( ((struct complex_double*)Y)[i+6] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+6] ) );
      ((struct complex_double*)Y)[i+7] = addition_cd( ((struct complex_double*)Y)[i+7] , multiplication_cd( *((struct complex_double*)a) , ((struct complex_double*)X)[i+7] ) );

    }

  return ;
}

void mncblas_zaxpy_vector(const int N, const void* a, const void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i ;

	double *tab_X = (double*)X;
	__m128d v1,v2,v3;

	double tab_a[2];
	tab_a[0]=((struct complex_double*)a)->real;
	tab_a[1]=((struct complex_double*)a)->imaginary;

	double tab_ai[2];
	tab_ai[0]=((struct complex_double*)a)->imaginary;
	tab_ai[1]=((struct complex_double*)a)->real;

#pragma omp parallel for schedule(static) private(v1,v2,v3)
  for (i=0; i < 2*N ; i += 2*incX)
    {
			v1 = _mm_load_pd(&tab_X[i]);

			v2 = _mm_load_pd(tab_a);
			v3 = _mm_mul_pd(v1,v2);

			((double*)Y) [i] += (v3[0] - v3[1]);

			v2 = _mm_load_pd(tab_ai);
			v3 = _mm_mul_pd(v1,v2);

			((double*)Y) [i+1] += (v3[0] + v3[1]);
    }

  return ;
}

