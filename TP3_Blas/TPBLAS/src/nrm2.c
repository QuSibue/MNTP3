
#include "mnblas.h"
#include "complex.h"
#include "math.h"
#include <omp.h>
#include <x86intrin.h>
#include <stdio.h>

float mncblas_snrm2(const int N,const float *X,const int incX)
{
  register unsigned int i = 0 ;
  float res=0;

  for (; (i < N) ; i += incX)
    {
      res+=X[i]*X[i];
    }
    return sqrt(res);
}

float mncblas_snrm2_static(const int N,const float *X,const int incX)
{
  register unsigned int i = 0 ;
  float res=0;
#pragma omp parallel for schedule(static) reduction (+:res)
  for (i=0; i < N ; i += 8*incX)
    {
      res+=X[i]*X[i];
      res+=X[i+1]*X[i+1];
      res+=X[i+2]*X[i+2];
      res+=X[i+3]*X[i+3];
      res+=X[i+4]*X[i+4];
      res+=X[i+5]*X[i+5];
      res+=X[i+6]*X[i+6];
      res+=X[i+7]*X[i+7];
    }
    return sqrt(res);
}

float mncblas_snrm2_vector(const int N,const float *X,const int incX)
{
  register unsigned int i = 0 ;
  float res=0;
	__m128 v1,v2;
#pragma omp parallel for schedule(static) reduction (+:res) private(v1,v2)
  for (i=0; i < N ; i += 4*incX)
    {
			v1 = _mm_load_ps(&X[i]);
			v2 = _mm_load_ps(&X[i]);
			v1 = _mm_mul_ps(v1,v2);
      res+=v1[0]+v1[1]+v1[2]+v1[3];
    }
    return sqrt(res);
}


double mncblas_dnrm2(const int N,const double *X,const int incX)
{
  register unsigned int i = 0 ;
  double res=0;

  for (; (i < N) ; i += incX)
    {
      res+=X[i]*X[i];
    }
    return sqrt(res);
}

double mncblas_dnrm2_static(const int N,const double *X,const int incX)
{
  register unsigned int i = 0 ;
  double res=0;
#pragma omp parallel for schedule(static) reduction (+:res)
  for (i=0; i < N ; i += 8*incX)
    {
      res+=X[i]*X[i];
      res+=X[i+1]*X[i+1];
      res+=X[i+2]*X[i+2];
      res+=X[i+3]*X[i+3];
      res+=X[i+4]*X[i+4];
      res+=X[i+5]*X[i+5];
      res+=X[i+6]*X[i+6];
      res+=X[i+7]*X[i+7];
    }
    return sqrt(res);
}

double mncblas_dnrm2_vector(const int N,const double *X,const int incX)
{
  register unsigned int i = 0 ;
  double res=0;
	__m128d v1,v2;
#pragma omp parallel for schedule(static) reduction (+:res) private(v1,v2)
  for (i=0; i < N ; i += 2*incX)
    {
			v1 = _mm_load_pd(&X[i]);
			v2 = _mm_load_pd(&X[i]);
			v1 = _mm_mul_pd(v1,v2);
      res+=v1[0]+v1[1];
    }
    return sqrt(res);
}

float mncblas_scnrm2(const int N,const void *X,const int incX)
{
  register unsigned int i = 0 ;
  float res=0;

  for (; (i < N) ; i += incX)
    {
      res +=  ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)X)[i].real )  +
	            ( ((struct complex_simple*)X)[i].imaginary) * ((struct complex_simple*)X)[i].imaginary;
    }
  return sqrt(res);
}

float mncblas_scnrm2_vector(const int N,const void *X,const int incX)
{
  register unsigned int i = 0 ;
  float res=0;
	__m128 v1,v2;

	float *tabfloat = ((float*)X);

#pragma omp parallel for schedule(static) reduction (+:res) private(v1,v2)
  for (i=0; i < 2*N ; i += 4*incX)
    {
			v1 = _mm_load_ps( &tabfloat[i]);
			v2 = _mm_load_ps( &tabfloat[i]);
			v1 = _mm_mul_ps(v1,v2);
      res+=v1[0]+v1[1]+v1[2]+v1[3];
    }
  return sqrt(res);
}


float mncblas_scnrm2_static(const int N,const void *X,const int incX)
{
  register unsigned int i = 0 ;
  float res=0;

#pragma omp parallel for schedule(static) reduction (+:res)
  for (i=0; i < N ; i += 8*incX)
    {
      res +=  ( ((struct complex_simple*)X)[i].real * ((struct complex_simple*)X)[i].real )  +
	            ( ((struct complex_simple*)X)[i].imaginary) * ((struct complex_simple*)X)[i].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+1].real * ((struct complex_simple*)X)[i+1].real )  +
							( ((struct complex_simple*)X)[i+1].imaginary) * ((struct complex_simple*)X)[i+1].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+2].real * ((struct complex_simple*)X)[i+2].real )  +
							( ((struct complex_simple*)X)[i+2].imaginary) * ((struct complex_simple*)X)[i+2].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+3].real * ((struct complex_simple*)X)[i+3].real )  +
							( ((struct complex_simple*)X)[i+3].imaginary) * ((struct complex_simple*)X)[i+3].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+4].real * ((struct complex_simple*)X)[i+4].real )  +
							( ((struct complex_simple*)X)[i+4].imaginary) * ((struct complex_simple*)X)[i+4].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+5].real * ((struct complex_simple*)X)[i+5].real )  +
							( ((struct complex_simple*)X)[i+5].imaginary) * ((struct complex_simple*)X)[i+5].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+6].real * ((struct complex_simple*)X)[i+6].real )  +
							( ((struct complex_simple*)X)[i+6].imaginary) * ((struct complex_simple*)X)[i+6].imaginary;

			res +=  ( ((struct complex_simple*)X)[i+7].real * ((struct complex_simple*)X)[i+7].real )  +
							( ((struct complex_simple*)X)[i+7].imaginary) * ((struct complex_simple*)X)[i+7].imaginary;



    }
  return sqrt(res);
}


double  mncblas_dznrm2(const int N,const void *X,const int incX)
{
  register unsigned int i = 0 ;
  double res=0;
  for (; (i < N) ; i += incX)
    {
	res += ( ((struct complex_double*)X)[i].real * ((struct complex_double*)X)[i].real )  +
                     ( ((struct complex_double*)X)[i].imaginary) * ((struct complex_double*)X)[i].imaginary;
    }
  return sqrt(res);
}

double  mncblas_dznrm2_vector(const int N,const void *X,const int incX)
{
  register unsigned int i = 0 ;
  double res=0;
	__m128d v1,v2;
	double *doubletab = ((double*)X);
#pragma omp parallel for schedule(static) reduction (+:res) private(v1,v2)
  for (i=0; i < 2*N ; i += 2*incX)
    {
			v1 = _mm_load_pd(&doubletab[i]);
			v2 = _mm_load_pd(&doubletab[i]);
			v1 = _mm_mul_pd(v1,v2);
      res+=v1[0]+v1[1];
    }
  return sqrt(res);
}


double  mncblas_dznrm2_static(const int N,const void *X,const int incX)
{
  register unsigned int i = 0 ;
  double res=0;

#pragma omp parallel for schedule(static) reduction (+:res)
  for (i=0; i < N ; i += 8*incX)
    {
			res += ( ((struct complex_double*)X)[i].real * ((struct complex_double*)X)[i].real )  +
				     ( ((struct complex_double*)X)[i].imaginary) * ((struct complex_double*)X)[i].imaginary;

			res += ( ((struct complex_double*)X)[i+1].real * ((struct complex_double*)X)[i+1].real )  +
				     ( ((struct complex_double*)X)[i+1].imaginary) * ((struct complex_double*)X)[i+1].imaginary;

			res += ( ((struct complex_double*)X)[i+2].real * ((struct complex_double*)X)[i+2].real )  +
				     ( ((struct complex_double*)X)[i+2].imaginary) * ((struct complex_double*)X)[i+2].imaginary;

			res += ( ((struct complex_double*)X)[i+3].real * ((struct complex_double*)X)[i+3].real )  +
				     ( ((struct complex_double*)X)[i+3].imaginary) * ((struct complex_double*)X)[i+3].imaginary;

			res += ( ((struct complex_double*)X)[i+4].real * ((struct complex_double*)X)[i+4].real )  +
				     ( ((struct complex_double*)X)[i+4].imaginary) * ((struct complex_double*)X)[i+4].imaginary;

			res += ( ((struct complex_double*)X)[i+5].real * ((struct complex_double*)X)[i+5].real )  +
				     ( ((struct complex_double*)X)[i+5].imaginary) * ((struct complex_double*)X)[i+5].imaginary;

			res += ( ((struct complex_double*)X)[i+6].real * ((struct complex_double*)X)[i+6].real )  +
				     ( ((struct complex_double*)X)[i+6].imaginary) * ((struct complex_double*)X)[i+6].imaginary;

			res += ( ((struct complex_double*)X)[i+7].real * ((struct complex_double*)X)[i+7].real )  +
				     ( ((struct complex_double*)X)[i+7].imaginary) * ((struct complex_double*)X)[i+7].imaginary;

    }
  return sqrt(res);
}
