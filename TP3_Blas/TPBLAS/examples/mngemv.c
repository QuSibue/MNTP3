#include <stdio.h>
#include <cblas.h>

#include "mnblas.h"
#include "complex.h"
#include "fonctions_test.h"

/*
  Mesure des cycles
*/

#include <omp.h>
#include <x86intrin.h>

//===================================================DEFINITION=============================================================================================//

mfloat mvecA,blmvecA;
vfloat vecX,blvecX,vecY,blvecY;

mdouble mvecdA,blmvecdA;
vdouble vecdX,vecdY,blvecdX,blvecdY;


mcsimple mveccsA,blmveccsA;
vcsimple veccsX,veccsY,blveccsX,blveccsY;
struct complex_simple alphacs,betacs;


mcdouble mveccdA,blmveccdA;
vcdouble veccdX,veccdY,blveccdX,blveccdY;
struct complex_double alphacd,betacd;

double m_Flops;



//=======================================================================================================================================================//


int main (int argc, char **argv)
{
 unsigned long long start, end ;
 unsigned long long residu ;

 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;


//====================================================vecteur float===========================================================//
  vector_Minit (mvecA, 1.0);
  vector_init ( vecX,2.0);
	vector_init ( vecY,3.0);

  vector_Minit (blmvecA, 1.0) ;
  vector_init ( blvecX,2.0);
	vector_init ( blvecY,3.0);


printf("=========================VECTEUR FLOAT================================\n");

  start = _rdtsc () ;
     mncblas_sgemv (MNCblasRowMajor,MNCblasNoTrans,M,N,1.0,blmvecA,M,blvecX,1,2.0,blvecY,1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm nombre de cycles cblas: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     mncblas_sgemv_static(MNCblasRowMajor,MNCblasNoTrans,M,N,1.0,mvecA,M,vecX,1,2.0,vecY,1);
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm_openmp: nombre de cycles: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflop : %f \n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurFloat(vecY,N,blvecY,N)){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }




//====================================================vecteur double===========================================================//
  vector_Minit_double (mvecdA, 1.0);
  vector_init_double ( vecdX,2.0);
	vector_init_double ( vecdY,3.0);

  vector_Minit_double (blmvecdA, 1.0) ;
  vector_init_double ( blvecdX,2.0);
	vector_init_double ( blvecdY,3.0);


printf("=========================VECTEUR DOUBLE================================\n");

  start = _rdtsc () ;
     mncblas_dgemv (MNCblasRowMajor,MNCblasNoTrans,M,N,1.0,blmvecdA,M,blvecdX,1,2.0,blvecdY,1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm nombre de cycles cblas: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     mncblas_dgemv_static(MNCblasRowMajor,MNCblasNoTrans,M,N,1.0,mvecdA,M,vecdX,1,2.0,vecdY,1);
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm_openmp: nombre de cycles: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflop : %f \n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurDouble(vecdY,N,blvecdY,N)){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



//====================================================vecteur complex_simple===========================================================//

struct complex_simple acs;acs.real=1.0;acs.imaginary=2.0;
struct complex_simple bcs;bcs.real=2.0;bcs.imaginary=4.0;
struct complex_simple ccs;ccs.real=3.0;ccs.imaginary=6.0;


  vector_Minit_csimple (mveccsA, acs);
  vector_init_csimple ( veccsX,bcs);
	vector_init_csimple ( veccsY,ccs);

  vector_Minit_csimple (blmveccsA, acs) ;
  vector_init_csimple ( blveccsX,bcs);
	vector_init_csimple ( blveccsY,ccs);

	alphacs.real=1.0;alphacs.imaginary=2.0;
	betacs.real=2.0;betacs.imaginary=3.0;

printf("=========================VECTEUR COMPLEX_SIMPLE================================\n");

  start = _rdtsc () ;
     mncblas_cgemv (MNCblasRowMajor,MNCblasNoTrans,M,N,&alphacs,blmveccsA,M,blveccsX,1,&betacs,blveccsY,1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm nombre de cycles cblas: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     mncblas_cgemv_static(MNCblasRowMajor,MNCblasNoTrans,M,N,&alphacs,mveccsA,M,veccsX,1,&betacs,veccsY,1);
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm_openmp: nombre de cycles: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflop : %f \n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurCS(veccsY,N,blveccsY,N)){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



//====================================================vecteur complex_double===========================================================//

struct complex_double acd;acd.real=1.0;acd.imaginary=2.0;
struct complex_double bcd;bcd.real=2.0;bcd.imaginary=4.0;
struct complex_double ccd;ccd.real=3.0;ccd.imaginary=6.0;


  vector_Minit_cdouble (mveccdA, acd);
  vector_init_cdouble ( veccdX,bcd);
	vector_init_cdouble ( veccdY,ccd);

  vector_Minit_cdouble (blmveccdA, acd) ;
  vector_init_cdouble ( blveccdX,bcd);
	vector_init_cdouble ( blveccdY,ccd);

	alphacd.real=1.0;alphacd.imaginary=2.0;
	betacd.real=2.0;betacd.imaginary=3.0;

printf("=========================VECTEUR COMPLEX_DOUBLE================================\n");

  start = _rdtsc () ;
     mncblas_zgemv (MNCblasRowMajor,MNCblasNoTrans,M,N,&alphacd,blmveccdA,M,blveccdX,1,&betacd,blveccdY,1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm nombre de cycles cblas: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     mncblas_zgemv_static(MNCblasRowMajor,MNCblasNoTrans,M,N,&alphacd,mveccdA,M,veccdX,1,&betacd,veccdY,1);
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,31*M*N + 61 *M*N,end-start-residu);
  printf ("mncblas_sgemm_openmp: nombre de cycles: %Ld \n", end-start-residu) ;
	printf ("resultat en Gflop : %f \n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurCD(veccdY,N,blveccdY,N)){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



}
