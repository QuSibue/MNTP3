#include "mnblas.h"
#include "complex.h"

void mncblas_saxpy(const int N, const float a, const float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      Y [j] = a* X[i] + Y[j] ;
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

void mncblas_caxpy(const int N, const void* a, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      float realTamp = ((struct complex_simple*)a)->real * ((struct complex_simple*)X)[i].real - ((struct complex_simple*)a)->imaginary * ((struct complex_simple*)X)[i].imaginary;
      float imaTamp = ((struct complex_simple*)a)->real * ((struct complex_simple*)X)[i].imaginary + ((struct complex_simple*)a)->imaginary * ((struct complex_simple*)X)[i].real;
      ((struct complex_simple*)Y)[j].real =  realTamp + ((struct complex_simple*)Y)[j].real;
      ((struct complex_simple*)Y)[j].imaginary = imaTamp + ((struct complex_simple*)Y)[j].imaginary;
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
      float realTamp = ((struct complex_double*)a)->real * ((struct complex_double*)X)[i].real - ((struct complex_double*)a)->imaginary * ((struct complex_double*)X)[i].imaginary;
      float imaTamp = ((struct complex_double*)a)->real * ((struct complex_double*)X)[i].imaginary + ((struct complex_double*)a)->imaginary * ((struct complex_double*)X)[i].real;
      ((struct complex_double*)Y)[j].real =  realTamp + ((struct complex_double*)Y)[j].real;
      ((struct complex_double*)Y)[j].imaginary = imaTamp + ((struct complex_double*)Y)[j].imaginary;
    }

  return ;
}
