#include "mnblas.h"
#include "complex.h"




void mncblas_scopy(const int N, const float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i=0 , j=0; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      Y [j] = X [i] ;
    }

}

void mncblas_scopy_static(const int N, const float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i =0;
  register unsigned int j =0;
#pragma omp parallel for schedule(static)
  for (i=j=0; i < N ; i += 8*incX )
    {
      Y [i] = X [i] ;
      Y [i+1] = X [i+1] ;
      Y [i+2] = X [i+2] ;
      Y [i+3] = X [i+3] ;
      Y [i+4] = X [i+4] ;
      Y [i+5] = X [i+5] ;
      Y [i+6] = X [i+6] ;
      Y [i+7] = X [i+7] ;
    }
}


void mncblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      Y [j] = X [i] ;
    }

  return ;
}

void mncblas_dcopy_static(const int N, const double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i ;
  register unsigned int j ;
#pragma omp parallel for schedule(static)
  for (i=j=0; i < N ; i += 8*incX )
    {
      Y [i] = X [i] ;
      Y [i+1] = X [i+1] ;
      Y [i+2] = X [i+2] ;
      Y [i+3] = X [i+3] ;
      Y [i+4] = X [i+4] ;
      Y [i+5] = X [i+5] ;
      Y [i+6] = X [i+6] ;
      Y [i+7] = X [i+7] ;
    }

  return ;
}


void mncblas_ccopy(const int N, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      ((struct complex_simple*)Y)[j].real = ((struct complex_simple*)X)[i].real ;
      ((struct complex_simple*)Y)[j].imaginary = ((struct complex_simple*)X)[i].imaginary ;
    }

  return ;
}

void mncblas_ccopy_static(const int N, const void *X, const int incX,
                void *Y, const int incY)
{
  register unsigned int i;
  register unsigned int j ;
#pragma omp parallel for schedule(static)

  for (i=j=0; i < N ; i += 8*incX )
    {
      ((struct complex_simple*)Y)[i].real = ((struct complex_simple*)X)[i].real ;
      ((struct complex_simple*)Y)[i].imaginary = ((struct complex_simple*)X)[i].imaginary ;

      ((struct complex_simple*)Y)[i+1].real = ((struct complex_simple*)X)[i+1].real ;
      ((struct complex_simple*)Y)[i+1].imaginary = ((struct complex_simple*)X)[i+1].imaginary ;

      ((struct complex_simple*)Y)[i+2].real = ((struct complex_simple*)X)[i+2].real ;
      ((struct complex_simple*)Y)[i+2].imaginary = ((struct complex_simple*)X)[i+2].imaginary ;

      ((struct complex_simple*)Y)[i+3].real = ((struct complex_simple*)X)[i+3].real ;
      ((struct complex_simple*)Y)[i+3].imaginary = ((struct complex_simple*)X)[i+3].imaginary ;

      ((struct complex_simple*)Y)[i+4].real = ((struct complex_simple*)X)[i+4].real ;
      ((struct complex_simple*)Y)[i+4].imaginary = ((struct complex_simple*)X)[i+4].imaginary ;

      ((struct complex_simple*)Y)[i+5].real = ((struct complex_simple*)X)[i+5].real ;
      ((struct complex_simple*)Y)[i+5].imaginary = ((struct complex_simple*)X)[i+5].imaginary ;

      ((struct complex_simple*)Y)[i+6].real = ((struct complex_simple*)X)[i+6].real ;
      ((struct complex_simple*)Y)[i+6].imaginary = ((struct complex_simple*)X)[i+6].imaginary ;

      ((struct complex_simple*)Y)[i+7].real = ((struct complex_simple*)X)[i+7].real ;
      ((struct complex_simple*)Y)[i+7].imaginary = ((struct complex_simple*)X)[i+7].imaginary ;
    }

}


void mncblas_zcopy(const int N, const void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      ((struct complex_double*)Y)[j].real = ((struct complex_double*)X)[i].real ;
      ((struct complex_double*)Y)[j].imaginary = ((struct complex_double*)X)[i].imaginary ;
    }

}

void mncblas_zcopy_static(const int N, const void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i ;
  register unsigned int j ;
#pragma omp parallel for schedule(static)

  for (i=j=0; i < N ; i += 8*incX )
    {
      ((struct complex_double*)Y)[i].real = ((struct complex_double*)X)[i].real ;
      ((struct complex_double*)Y)[i].imaginary = ((struct complex_double*)X)[i].imaginary ;

      ((struct complex_double*)Y)[i+1].real = ((struct complex_double*)X)[i+1].real ;
      ((struct complex_double*)Y)[i+1].imaginary = ((struct complex_double*)X)[i+1].imaginary ;

      ((struct complex_double*)Y)[i+2].real = ((struct complex_double*)X)[i+2].real ;
      ((struct complex_double*)Y)[i+2].imaginary = ((struct complex_double*)X)[i+2].imaginary ;

      ((struct complex_double*)Y)[i+3].real = ((struct complex_double*)X)[i+3].real ;
      ((struct complex_double*)Y)[i+3].imaginary = ((struct complex_double*)X)[i+3].imaginary ;

      ((struct complex_double*)Y)[i+4].real = ((struct complex_double*)X)[i+4].real ;
      ((struct complex_double*)Y)[i+4].imaginary = ((struct complex_double*)X)[i+4].imaginary ;

      ((struct complex_double*)Y)[i+5].real = ((struct complex_double*)X)[i+5].real ;
      ((struct complex_double*)Y)[i+5].imaginary = ((struct complex_double*)X)[i+5].imaginary ;

      ((struct complex_double*)Y)[i+6].real = ((struct complex_double*)X)[i+6].real ;
      ((struct complex_double*)Y)[i+6].imaginary = ((struct complex_double*)X)[i+6].imaginary ;

      ((struct complex_double*)Y)[i+7].real = ((struct complex_double*)X)[i+7].real ;
      ((struct complex_double*)Y)[i+7].imaginary = ((struct complex_double*)X)[i+7].imaginary ;

    }
}

