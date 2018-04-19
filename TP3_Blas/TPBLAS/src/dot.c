#include "mnblas.h"
#include "complex.h"
#include <omp.h>
#include <x86intrin.h>

float mncblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float dot = 0.0 ;
//#pragma omp parallel for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
	//#pragma omp critical
      dot = dot + X [i] * Y [j] ;
    }

  return dot ;
}

double mncblas_ddot(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double dot = 0.0 ;
//#pragma omp parallel for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
	//#pragma omp critical
      dot = dot + X [i] * Y [j] ;
    }
  return dot;
}

void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  ((struct complex_simple*)dotu)->real = 0;
  ((struct complex_simple*)dotu)->imaginary = 0;
//#pragma omp parallel for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
	//#pragma omp critical
      ((struct complex_simple*)dotu)->real +=( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[j].real ) - ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[j].imaginary) ) ;

	//#pragma omp critical
      ((struct complex_simple*)dotu)->imaginary += ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[j].imaginary ) + ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[j].real ) );
    }

  return ;
}

void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  ((struct complex_simple*)dotc)->real = 0;
  ((struct complex_simple*)dotc)->imaginary = 0;
//#pragma omp parallel for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
	//#pragma omp critical
      ((struct complex_simple*)dotc)->real +=  ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[j].real ) + ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[j].imaginary) ) ;
	//#pragma omp critical
      ((struct complex_simple*)dotc)->imaginary += ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[j].imaginary ) - ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[j].real ) );
    }

  return ;
}

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  ((struct complex_double*)dotu)->real = 0;
  ((struct complex_double*)dotu)->imaginary = 0;
//#pragma omp parallel for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
	//#pragma omp critical
      ((struct complex_double*)dotu)->real +=  ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[j].real ) - ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[j].imaginary) ) ;
	//#pragma omp critical
      ((struct complex_double*)dotu)->imaginary += ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[j].imaginary ) + ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[j].real ) );
    }
  return ;
}

void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  ((struct complex_double*)dotc)->real = 0;
  ((struct complex_double*)dotc)->real = 0;
//#pragma omp parallel for
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
	//#pragma omp critical
      ((struct complex_double*)dotc)->real +=  ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[j].real ) + ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[j].imaginary) ) ;
	//#pragma omp critical
      ((struct complex_double*)dotc)->imaginary += ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[j].imaginary ) - ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[j].real ) );
    }
  return ;
}
