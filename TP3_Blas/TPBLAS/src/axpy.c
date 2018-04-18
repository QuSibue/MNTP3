#include "mnblas.h"
#include "complex.h"

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
  register unsigned int j;

  for (i=0; i < N ; i += 8*incX)
    {
      Y [j] += a* X[i];
			Y [j+1] += a* X[i+1];
			Y [j+2] += a* X[i+2];
			Y [j+3] += a* X[i+3];
			Y [j+4] += a* X[i+4];
			Y [j+5] += a* X[i+5];
			Y [j+6] += a* X[i+6];
			Y [j+7] += a* X[i+7];
			
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
  register unsigned int j = 0 ;
  for (i=0; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
        Y [j] += a* X[i];
			Y [j+1] += a* X[i+1];
			Y [j+2] += a* X[i+2];
			Y [j+3] += a* X[i+3];
			Y [j+4] += a* X[i+4];
			Y [j+5] += a* X[i+5];
			Y [j+6] += a* X[i+6];
			Y [j+7] += a* X[i+7];
    }

  return ;
}



void mncblas_caxpy(const int N, const void* a, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
	float realTamp;
	float imaTamp;
//#pragma omp parallel for private(realTamp,imaTamp)
  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      realTamp = ( ((struct complex_simple*)a)->real * ((struct complex_simple*)X)[i].real - ((struct complex_simple*)a)->imaginary * ((struct complex_simple*)X)[i].imaginary );
      imaTamp = ((struct complex_simple*)a)->real * ((struct complex_simple*)X)[i].imaginary + ((struct complex_simple*)a)->imaginary * ((struct complex_simple*)X)[i].real;
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

	double realTamp;
	double imaTamp;
//#pragma omp parallel for private(realTamp,imaTamp)
  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      realTamp = ((struct complex_double*)a)->real * ((struct complex_double*)X)[i].real - ((struct complex_double*)a)->imaginary * ((struct complex_double*)X)[i].imaginary;
      imaTamp = ((struct complex_double*)a)->real * ((struct complex_double*)X)[i].imaginary + ((struct complex_double*)a)->imaginary * ((struct complex_double*)X)[i].real;
      ((struct complex_double*)Y)[j].real =  realTamp + ((struct complex_double*)Y)[j].real;
      ((struct complex_double*)Y)[j].imaginary = imaTamp + ((struct complex_double*)Y)[j].imaginary;
    }

  return ;
}
