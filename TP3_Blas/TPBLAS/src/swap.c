#include "mnblas.h"
#include "complex.h"

void mncblas_sswap(const int N, float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float save ;
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_sswap_static(const int N, float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i ;
  register float save ;

#pragma omp parallel for schedule(static) private(save)
  for (i=0; i < N; i += incX)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;

      save = Y [i+1] ;
      Y [i+1] = X [i+1] ;
      X [i+1] = save ;

      save = Y [i+2] ;
      Y [i+2] = X [i+2] ;
      X [i+2] = save ;

      save = Y [i+2] ;
      Y [i+2] = X [i+2] ;
      X [i] = save ;

      save = Y [i+4] ;
      Y [i+4] = X [i+4] ;
      X [i+4] = save ;

      save = Y [i+5] ;
      Y [i+5] = X [i+5] ;
      X [i+5] = save ;

      save = Y [i+6] ;
      Y [i+6] = X [i+6] ;
      X [i+6] = save ;

      save = Y [i+7] ;
      Y [i+7] = X [i+7] ;
      X [i+7] = save ;
    }

  return ;
}


void mncblas_dswap(const int N, double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double save ;
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_dswap_static(const int N, double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i;
  register double save ;

#pragma omp parallel for schedule(static) private(save)
  for (i=0; i < N; i += incX)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;

      save = Y [i+1] ;
      Y [i+1] = X [i+1] ;
      X [i+1] = save ;

      save = Y [i+2] ;
      Y [i+2] = X [i+2] ;
      X [i+2] = save ;

      save = Y [i+2] ;
      Y [i+2] = X [i+2] ;
      X [i] = save ;

      save = Y [i+4] ;
      Y [i+4] = X [i+4] ;
      X [i+4] = save ;

      save = Y [i+5] ;
      Y [i+5] = X [i+5] ;
      X [i+5] = save ;

      save = Y [i+6] ;
      Y [i+6] = X [i+6] ;
      X [i+6] = save ;

      save = Y [i+7] ;
      Y [i+7] = X [i+7] ;
      X [i+7] = save ;
    }

  return ;
}

void mncblas_cswap(const int N, void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register struct complex_simple save ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = ((struct complex_simple *)Y) [j] ;
      ((struct complex_simple *)Y) [j] = ((struct complex_simple *)X) [i] ;
      ((struct complex_simple *)X) [i] = save ;
    }
  return ;
}

void mncblas_cswap_static(const int N, void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register struct complex_simple save ;

#pragma omp parallel for schedule(static) private(save)
  for (i=0; i < N ; i += incX)
    {
      save = ((struct complex_simple *)Y) [i] ;
      ((struct complex_simple *)Y) [i] = ((struct complex_simple *)X) [i] ;
      ((struct complex_simple *)X) [i] = save ;

      save = ((struct complex_simple *)Y) [i+1] ;
      ((struct complex_simple *)Y) [i+1] = ((struct complex_simple *)X) [i+1] ;
      ((struct complex_simple *)X) [i+1] = save ;

      save = ((struct complex_simple *)Y) [i+2] ;
      ((struct complex_simple *)Y) [i+2] = ((struct complex_simple *)X) [i+2] ;
      ((struct complex_simple *)X) [i+2] = save ;

      save = ((struct complex_simple *)Y) [i+3] ;
      ((struct complex_simple *)Y) [i+3] = ((struct complex_simple *)X) [i+3] ;
      ((struct complex_simple *)X) [i+3] = save ;

      save = ((struct complex_simple *)Y) [i+4] ;
      ((struct complex_simple *)Y) [i+4] = ((struct complex_simple *)X) [i+4] ;
      ((struct complex_simple *)X) [i+4] = save ;

      save = ((struct complex_simple *)Y) [i+5] ;
      ((struct complex_simple *)Y) [i+5] = ((struct complex_simple *)X) [i+5] ;
      ((struct complex_simple *)X) [i+5] = save ;

      save = ((struct complex_simple *)Y) [i+6] ;
      ((struct complex_simple *)Y) [i+6] = ((struct complex_simple *)X) [i+6] ;
      ((struct complex_simple *)X) [i+6] = save ;

      save = ((struct complex_simple *)Y) [i+7] ;
      ((struct complex_simple *)Y) [i+7] = ((struct complex_simple *)X) [i+7] ;
      ((struct complex_simple *)X) [i+7] = save ;


    }
  return ;
}


void mncblas_zswap(const int N, void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register struct complex_double save ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
  {
    save = ((struct complex_double *)Y) [j] ;
    ((struct complex_double *)Y) [j] = ((struct complex_double *)X) [i] ;
    ((struct complex_double *)X) [i] = save ;
  }
  return ;
}

void mncblas_zswap_static(const int N, void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register struct complex_double save ;

#pragma omp parallel for schedule(static) private(save)
  for (i=0; i < N; i += incX)
  {
    save = ((struct complex_double *)Y) [i] ;
    ((struct complex_double *)Y) [i] = ((struct complex_double *)X) [i] ;
    ((struct complex_double *)X) [i] = save ;

    save = ((struct complex_double *)Y) [i+1] ;
    ((struct complex_double *)Y) [i+1] = ((struct complex_double *)X) [i+1] ;
    ((struct complex_double *)X) [i+1] = save ;

    save = ((struct complex_double *)Y) [i+2] ;
    ((struct complex_double *)Y) [i+2] = ((struct complex_double *)X) [i+2] ;
    ((struct complex_double *)X) [i+2] = save ;

    save = ((struct complex_double *)Y) [i+3] ;
    ((struct complex_double *)Y) [i+3] = ((struct complex_double *)X) [i+3] ;
    ((struct complex_double *)X) [i+3] = save ;

    save = ((struct complex_double *)Y) [i+4] ;
    ((struct complex_double *)Y) [i+4] = ((struct complex_double *)X) [i+4] ;
    ((struct complex_double *)X) [i+4] = save ;

    save = ((struct complex_double *)Y) [i+5] ;
    ((struct complex_double *)Y) [i+5] = ((struct complex_double *)X) [i+5] ;
    ((struct complex_double *)X) [i+5] = save ;

    save = ((struct complex_double *)Y) [i+6] ;
    ((struct complex_double *)Y) [i+6] = ((struct complex_double *)X) [i+6] ;
    ((struct complex_double *)X) [i+6] = save ;

    save = ((struct complex_double *)Y) [i+7] ;
    ((struct complex_double *)Y) [i+7] = ((struct complex_double *)X) [i+7] ;
    ((struct complex_double *)X) [i+7] = save ;
  }
  return ;
}

