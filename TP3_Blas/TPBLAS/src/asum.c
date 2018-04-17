#include "mnblas.h"
#include "complex.h"
#include "math.h"

float mncblas_sasum(const int N, const float *X, const int incX)
{
  register unsigned int i = 0 ;
  float resultat=0;
  for (; (i < N) ; i += incX)
    {
      resultat += fabs(X[i]) ;
    }
  return resultat;
}


float mncblas_sasum_static(const int N, const float* X,
                const int incX)
{
  register unsigned int i;

  float resultat = 0;
#pragma omp parallel for schedule(static) reduction (+:resultat)
  for (i=0; i < N; i += 8*incX) {
    resultat += fabs(X[i]);
    resultat += fabs(X[i+1]);
    resultat += fabs(X[i+2]);
    resultat += fabs(X[i+3]);
    resultat += fabs(X[i+4]);
    resultat += fabs(X[i+5]);
    resultat += fabs(X[i+6]);
    resultat += fabs(X[i+7]);
  }
  return resultat;
}

double mncblas_dasum(const int N, const double *X, const int incX)
{
  register unsigned int i = 0 ;
  double resultat=0;

  for (; (i < N) ; i += incX)
    {
      resultat += fabs(X[i]);
    }

  return resultat;
}



double mncblas_dasum_static(const int N, const double *X,
                const int incX)
{
  register unsigned int i;

  double res = 0.0;
#pragma omp parallel for schedule(static) reduction (+:res)
  for (i=0; i < N; i += 8*incX) {
    res += fabs(X[i]);
    res += fabs(X[i+1]);
    res += fabs(X[i+2]);
    res += fabs(X[i+3]);
    res += fabs(X[i+4]);
    res += fabs(X[i+5]);
    res += fabs(X[i+6]);
    res += fabs(X[i+7]);
  }
  return res; }

float mncblas_scasum(const int N, const void *X, const int incX)
{
  register unsigned int i = 0 ;
  float resultat = 0;


  for (; (i < N) ; i += incX)
    {
      resultat += fabs(((struct complex_simple*)X)[i].real) + fabs(((struct complex_simple*)X) [i].imaginary );
    }

  return resultat;
}

double mncblas_dzasum(const int N, const void *X, const int incX)
{
  register unsigned int i = 0 ;
  double resultat = 0;


  for (; (i < N); i += incX)
    {
      resultat += fabs( ((struct complex_double*)X)[i].real ) + fabs( ((struct complex_double*)X)[i].imaginary );
    }

  return resultat;

}
