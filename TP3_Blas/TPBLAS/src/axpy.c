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
