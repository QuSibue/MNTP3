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
vfloat vec1,blvec1,vvec1;
float resultatf,resultatcs;
float resultatf2,resultatcs2;
float vresultatf,vresultatcs;

vdouble vecd1,blvecd1,vvecd1;
double resultatd,resultatcd;
double resultatd2,resultatcd2;
double vresultatd,vresultatcd;


vcsimple veccs1,blveccs1,veccs2,blveccs2;
vcdouble veccd1,blveccd1,veccd2,blveccd1;
double m_Flops;


//=======================================================================================================================================================//

int main (int argc, char **argv)
{
 unsigned long long start, end ;
 unsigned long long residu ;

  vector_init (vec1, 1.0) ;
  vector_init_double(vecd1,2.0);
  struct complex_simple x;
  x.real = 2.0;
  x.imaginary = 3.0;
  vector_init_csimple(veccs1,x);
	struct complex_double y;
  y.real = 4.0;
  y.imaginary = 5.0;
  vector_init_cdouble(veccd1,y);

 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;


//====================================================vecteur float===========================================================//


printf("=========================VECTEUR FLOAT================================\n");

  start = _rdtsc () ;
     resultatf=cblas_snrm2(VECSIZE, vec1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);

  printf ("mncblas_snrm2 nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatf);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");


  start = _rdtsc () ;
     resultatf2=mncblas_snrm2_static(VECSIZE, vec1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);

  printf ("mncblas_snrm2_openmp nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatf2);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

	start = _rdtsc () ;
     vresultatf=mncblas_snrm2_vector(VECSIZE, vec1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);

  printf ("mncblas_snrm2_vector nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",vresultatf);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  if(resultatf == resultatf2 && resultatf == vresultatf){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



  printf("======================================================================\n\n");

//============================================================================================================================//


//====================================================vecteur double===========================================================//


printf("=========================VECTEUR Double================================\n");


  start = _rdtsc () ;
     resultatd = mncblas_dnrm2 (VECSIZE, vecd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatd);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatd2 = mncblas_dnrm2_static (VECSIZE, vecd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dasum_openmp nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatd2);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     vresultatd = mncblas_dnrm2_vector (VECSIZE, vecd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dasum_vectorisé nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",vresultatd);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  if(resultatd == resultatd2 && resultatd == vresultatd){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



  printf("======================================================================\n\n");


//============================================================================================================================//

//====================================================vecteur complex_simple===========================================================//


  printf("=========================VECTEUR COMPLEXES SIMPLES================================\n");

  start = _rdtsc () ;
     resultatcs = mncblas_scnrm2 (VECSIZE, veccs1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_scnrm2 nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcs);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatcs2 = mncblas_scnrm2_static (VECSIZE, veccs1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_scnrm2_openmp nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcs2);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

	start = _rdtsc () ;
     vresultatcs = mncblas_scnrm2_vector (VECSIZE, veccs1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_scnrm2_vector nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",vresultatcs);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  if(resultatcs2 == resultatcs && vresultatcs == resultatcs){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



  printf("=============================================================================\n\n");


//=====================================================================================================================================//


//====================================================vecteur complex_double===========================================================//
  

printf("=========================VECTEUR COMPLEXES DOUBLES================================\n");
  start = _rdtsc () ;
     resultatcd = mncblas_dznrm2 (VECSIZE, veccd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dznrm2 nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcd);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatcd2 = mncblas_dznrm2_static (VECSIZE, veccd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dznrm2_openmp nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcd2);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     vresultatcd = mncblas_dznrm2_static (VECSIZE, veccd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dznrm2_vector nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",vresultatcd);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");

  if(resultatcd == resultatcd2 && resultatcd == vresultatcd){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }


  printf("=============================================================================\n\n");
//=====================================================================================================================================//


}
