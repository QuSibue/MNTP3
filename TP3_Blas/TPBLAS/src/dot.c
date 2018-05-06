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
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot += X [i] * Y [j] ;
    }

  return dot ;
}

float mncblas_sdot_static(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
  register unsigned int i;
  register float dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction(+:dot)
  for (i=0; i < N ; i += 8*incX)
    {
      dot += X [i] * Y [i] ;
      dot += X [i+1] * Y [i+1] ;
      dot += X [i+2] * Y [i+2] ;
      dot += X [i+3] * Y [i+3] ;
      dot += X [i+4] * Y [i+4] ;
      dot += X [i+5] * Y [i+5] ;
      dot += X [i+6] * Y [i+6] ;
      dot += X [i+7] * Y [i+7] ;
    }

  return dot ;
}

double mncblas_ddot(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double dot = 0.0 ;
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot = dot + X [i] * Y [j] ;
    }
  return dot;
}

double mncblas_ddot_static(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register double dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction(+:dot)
  for (i=0; i < N; i += 8*incX)
    {
      dot += X [i] * Y [i] ;
      dot += X [i+1] * Y [i+1] ;
      dot += X [i+2] * Y [i+2] ;
      dot += X [i+3] * Y [i+3] ;
      dot += X [i+4] * Y [i+4] ;
      dot += X [i+5] * Y [i+5] ;
      dot += X [i+6] * Y [i+6] ;
      dot += X [i+7] * Y [i+7] ;
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
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      ((struct complex_simple*)dotu)->real +=( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[j].real ) - ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[j].imaginary) ) ;

      ((struct complex_simple*)dotu)->imaginary += ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[j].imaginary ) + ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[j].real ) );
    }

  return ;
}

void   mncblas_cdotu_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;

  float real =0;
  float imaginary = 0;

#pragma omp parallel for schedule(static) reduction(+: real , imaginary)
  for (i=0; i < N ; i += 8*incX)
    {
      real +=( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[i].real ) - ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[i].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[i].imaginary ) + ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[i].real ) );


      real +=( ( ((struct complex_simple*)X)[i+1].real * ((struct complex_simple*)Y)[i+1].real ) - ( ((struct complex_simple*)X)[i+1].imaginary * ((struct complex_simple*)Y)[i+1].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+1].real * ((struct complex_simple*)Y)[i+1].imaginary ) + ( ((struct complex_simple*)X)[i+1].imaginary * ((struct complex_simple*)Y)[i+1].real ) );


      real +=( ( ((struct complex_simple*)X)[i+2].real * ((struct complex_simple*)Y)[i+2].real ) - ( ((struct complex_simple*)X)[i+2].imaginary * ((struct complex_simple*)Y)[i+2].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+2].real * ((struct complex_simple*)Y)[i+2].imaginary ) + ( ((struct complex_simple*)X)[i+2].imaginary * ((struct complex_simple*)Y)[i+2].real ) );


      real +=( ( ((struct complex_simple*)X)[i+3].real * ((struct complex_simple*)Y)[i+3].real ) - ( ((struct complex_simple*)X)[i+3].imaginary * ((struct complex_simple*)Y)[i+3].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+3].real * ((struct complex_simple*)Y)[i+3].imaginary ) + ( ((struct complex_simple*)X)[i+3].imaginary * ((struct complex_simple*)Y)[i+3].real ) );


      real +=( ( ((struct complex_simple*)X)[i+4].real * ((struct complex_simple*)Y)[i+4].real ) - ( ((struct complex_simple*)X)[i+4].imaginary * ((struct complex_simple*)Y)[i+4].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+4].real * ((struct complex_simple*)Y)[i+4].imaginary ) + ( ((struct complex_simple*)X)[i+4].imaginary * ((struct complex_simple*)Y)[i+4].real ) );


      real +=( ( ((struct complex_simple*)X)[i+5].real * ((struct complex_simple*)Y)[i+5].real ) - ( ((struct complex_simple*)X)[i+5].imaginary * ((struct complex_simple*)Y)[i+5].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+5].real * ((struct complex_simple*)Y)[i+5].imaginary ) + ( ((struct complex_simple*)X)[i+5].imaginary * ((struct complex_simple*)Y)[i+5].real ) );


      real +=( ( ((struct complex_simple*)X)[i+6].real * ((struct complex_simple*)Y)[i+6].real ) - ( ((struct complex_simple*)X)[i+6].imaginary * ((struct complex_simple*)Y)[i+6].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+6].real * ((struct complex_simple*)Y)[i+6].imaginary ) + ( ((struct complex_simple*)X)[i+6].imaginary * ((struct complex_simple*)Y)[i+6].real ) );


      real +=( ( ((struct complex_simple*)X)[i+7].real * ((struct complex_simple*)Y)[i+7].real ) - ( ((struct complex_simple*)X)[i+7].imaginary * ((struct complex_simple*)Y)[i+7].imaginary) ) ;

      imaginary += ( ( ((struct complex_simple*)X)[i+7].real * ((struct complex_simple*)Y)[i+7].imaginary ) + ( ((struct complex_simple*)X)[i+7].imaginary * ((struct complex_simple*)Y)[i+7].real ) );

    }

    ((struct complex_simple*)dotu)->real = real;
    ((struct complex_simple*)dotu)->imaginary = imaginary;

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


void   mncblas_cdotc_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;

  float real =0;
  float imaginary = 0;

#pragma omp parallel for schedule(static) reduction(+:real ,imaginary)
  for (i=0; i < N; i += 8*incX)
    {
      real +=  ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[i].real ) + ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[i].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)Y)[i].imaginary ) - ( ((struct complex_simple*)X)[i].imaginary * ((struct complex_simple*)Y)[i].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+1].real * ((struct complex_simple*)Y)[i+1].real ) + ( ((struct complex_simple*)X)[i+1].imaginary * ((struct complex_simple*)Y)[i+1].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+1].real * ((struct complex_simple*)Y)[i+1].imaginary ) - ( ((struct complex_simple*)X)[i+1].imaginary * ((struct complex_simple*)Y)[i+1].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+2].real * ((struct complex_simple*)Y)[i+2].real ) + ( ((struct complex_simple*)X)[i+2].imaginary * ((struct complex_simple*)Y)[i+2].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+2].real * ((struct complex_simple*)Y)[i+2].imaginary ) - ( ((struct complex_simple*)X)[i+2].imaginary * ((struct complex_simple*)Y)[i+2].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+3].real * ((struct complex_simple*)Y)[i+3].real ) + ( ((struct complex_simple*)X)[i+3].imaginary * ((struct complex_simple*)Y)[i+3].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+3].real * ((struct complex_simple*)Y)[i+3].imaginary ) - ( ((struct complex_simple*)X)[i+3].imaginary * ((struct complex_simple*)Y)[i+3].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+4].real * ((struct complex_simple*)Y)[i+4].real ) + ( ((struct complex_simple*)X)[i+4].imaginary * ((struct complex_simple*)Y)[i+4].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+4].real * ((struct complex_simple*)Y)[i+4].imaginary ) - ( ((struct complex_simple*)X)[i+4].imaginary * ((struct complex_simple*)Y)[i+4].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+5].real * ((struct complex_simple*)Y)[i+5].real ) + ( ((struct complex_simple*)X)[i+5].imaginary * ((struct complex_simple*)Y)[i+5].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+5].real * ((struct complex_simple*)Y)[i+5].imaginary ) - ( ((struct complex_simple*)X)[i+5].imaginary * ((struct complex_simple*)Y)[i+5].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+6].real * ((struct complex_simple*)Y)[i+6].real ) + ( ((struct complex_simple*)X)[i+6].imaginary * ((struct complex_simple*)Y)[i+6].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+6].real * ((struct complex_simple*)Y)[i+6].imaginary ) - ( ((struct complex_simple*)X)[i+6].imaginary * ((struct complex_simple*)Y)[i+6].real ) );

      real +=  ( ( ((struct complex_simple*)X)[i+7].real * ((struct complex_simple*)Y)[i+7].real ) + ( ((struct complex_simple*)X)[i+7].imaginary * ((struct complex_simple*)Y)[i+7].imaginary) ) ;
      imaginary += ( ( ((struct complex_simple*)X)[i+7].real * ((struct complex_simple*)Y)[i+7].imaginary ) - ( ((struct complex_simple*)X)[i+7].imaginary * ((struct complex_simple*)Y)[i+7].real ) );
    }

    ((struct complex_simple*)dotc)->real = real;
    ((struct complex_simple*)dotc)->imaginary = imaginary;


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

void   mncblas_zdotu_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  double real =0;
  double imaginary = 0;

#pragma omp parallel for schedule(static) reduction(+:real ,imaginary)
  for (i=0; i < N ; i += 8*incX)
    {

      real +=  ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[i].real ) - ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[i].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[i].imaginary ) + ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[i].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+1].real * ((struct complex_double*)Y)[i+1].real ) - ( ((struct complex_double*)X)[i+1].imaginary * ((struct complex_double*)Y)[i+1].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+1].real * ((struct complex_double*)Y)[i+1].imaginary ) + ( ((struct complex_double*)X)[i+1].imaginary * ((struct complex_double*)Y)[i+1].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+2].real * ((struct complex_double*)Y)[i+2].real ) - ( ((struct complex_double*)X)[i+2].imaginary * ((struct complex_double*)Y)[i+2].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+2].real * ((struct complex_double*)Y)[i+2].imaginary ) + ( ((struct complex_double*)X)[i+2].imaginary * ((struct complex_double*)Y)[i+2].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+3].real * ((struct complex_double*)Y)[i+3].real ) - ( ((struct complex_double*)X)[i+3].imaginary * ((struct complex_double*)Y)[i+3].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+3].real * ((struct complex_double*)Y)[i+3].imaginary ) + ( ((struct complex_double*)X)[i+3].imaginary * ((struct complex_double*)Y)[i+3].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+4].real * ((struct complex_double*)Y)[i+4].real ) - ( ((struct complex_double*)X)[i+4].imaginary * ((struct complex_double*)Y)[i+4].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+4].real * ((struct complex_double*)Y)[i+4].imaginary ) + ( ((struct complex_double*)X)[i+4].imaginary * ((struct complex_double*)Y)[i+4].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+5].real * ((struct complex_double*)Y)[i+5].real ) - ( ((struct complex_double*)X)[i+5].imaginary * ((struct complex_double*)Y)[i+5].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+5].real * ((struct complex_double*)Y)[i+5].imaginary ) + ( ((struct complex_double*)X)[i+5].imaginary * ((struct complex_double*)Y)[i+5].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+6].real * ((struct complex_double*)Y)[i+6].real ) - ( ((struct complex_double*)X)[i+6].imaginary * ((struct complex_double*)Y)[i+6].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+6].real * ((struct complex_double*)Y)[i+6].imaginary ) + ( ((struct complex_double*)X)[i+6].imaginary * ((struct complex_double*)Y)[i+6].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+7].real * ((struct complex_double*)Y)[i+7].real ) - ( ((struct complex_double*)X)[i+7].imaginary * ((struct complex_double*)Y)[i+7].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+7].real * ((struct complex_double*)Y)[i+7].imaginary ) + ( ((struct complex_double*)X)[i+7].imaginary * ((struct complex_double*)Y)[i+7].real ) );
    }
	
    ((struct complex_double*)dotu)->real = real;
    ((struct complex_double*)dotu)->imaginary = imaginary;

  return ;
}



void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  ((struct complex_double*)dotc)->real = 0;
  ((struct complex_double*)dotc)->real = 0;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {

      ((struct complex_double*)dotc)->real +=  ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[j].real ) + ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[j].imaginary) ) ;

      ((struct complex_double*)dotc)->imaginary += ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[j].imaginary ) - ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[j].real ) );
    }
  return ;
}

void   mncblas_zdotc_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;


  double real =0;
  double imaginary = 0;

#pragma omp parallel for schedule(static) reduction(+:real ,imaginary)
  for (i=0; i < N; i += 8*incX)
    {
      real +=  ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[i].real ) + ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[i].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i].real * ((struct complex_double*)Y)[i].imaginary ) - ( ((struct complex_double*)X)[i].imaginary * ((struct complex_double*)Y)[i].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+1].real * ((struct complex_double*)Y)[i+1].real ) + ( ((struct complex_double*)X)[i+1].imaginary * ((struct complex_double*)Y)[i+1].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+1].real * ((struct complex_double*)Y)[i+1].imaginary ) - ( ((struct complex_double*)X)[i+1].imaginary * ((struct complex_double*)Y)[i+1].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+2].real * ((struct complex_double*)Y)[i+2].real ) + ( ((struct complex_double*)X)[i+2].imaginary * ((struct complex_double*)Y)[i+2].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+2].real * ((struct complex_double*)Y)[i+2].imaginary ) - ( ((struct complex_double*)X)[i+2].imaginary * ((struct complex_double*)Y)[i+2].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+3].real * ((struct complex_double*)Y)[i+3].real ) + ( ((struct complex_double*)X)[i+3].imaginary * ((struct complex_double*)Y)[i+3].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+3].real * ((struct complex_double*)Y)[i+3].imaginary ) - ( ((struct complex_double*)X)[i+3].imaginary * ((struct complex_double*)Y)[i+3].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+4].real * ((struct complex_double*)Y)[i+4].real ) + ( ((struct complex_double*)X)[i+4].imaginary * ((struct complex_double*)Y)[i+4].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+4].real * ((struct complex_double*)Y)[i+4].imaginary ) - ( ((struct complex_double*)X)[i+4].imaginary * ((struct complex_double*)Y)[i+4].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+5].real * ((struct complex_double*)Y)[i+5].real ) + ( ((struct complex_double*)X)[i+5].imaginary * ((struct complex_double*)Y)[i+5].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+5].real * ((struct complex_double*)Y)[i+5].imaginary ) - ( ((struct complex_double*)X)[i+5].imaginary * ((struct complex_double*)Y)[i+5].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+6].real * ((struct complex_double*)Y)[i+6].real ) + ( ((struct complex_double*)X)[i+6].imaginary * ((struct complex_double*)Y)[i+6].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+6].real * ((struct complex_double*)Y)[i+6].imaginary ) - ( ((struct complex_double*)X)[i+6].imaginary * ((struct complex_double*)Y)[i+6].real ) );

			real +=  ( ( ((struct complex_double*)X)[i+7].real * ((struct complex_double*)Y)[i+7].real ) + ( ((struct complex_double*)X)[i+7].imaginary * ((struct complex_double*)Y)[i+7].imaginary) ) ;
      imaginary += ( ( ((struct complex_double*)X)[i+7].real * ((struct complex_double*)Y)[i+7].imaginary ) - ( ((struct complex_double*)X)[i+7].imaginary * ((struct complex_double*)Y)[i+7].real ) );
    }
	
    ((struct complex_double*)dotc)->real = real;
    ((struct complex_double*)dotc)->imaginary = imaginary;
return;
}
