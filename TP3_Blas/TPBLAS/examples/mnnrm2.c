#include <stdio.h>
#include <cblas.h>

#include "mnblas.h"
#include "complex.h"
#include "fonctions_test.h"

/*
  Mesure des cycles
*/

#include <x86intrin.h>

//===================================================DEFINITION=============================================================================================//
vfloat vec1,blvec1,vec2,blvec2 ;
float resultatf,resultatcs;
float resultatf2,resultatcs2;
vdouble vecd1,blvecd1,vecd2,blvecd1;
double resultatd,resultatcd;
double resultatd2,resultatcd2;
vcsimple veccs1,blveccs1,veccs2,blveccs2;
vcdouble veccd1,blveccd1,veccd2,blveccd1;
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
  vector_init (vec1, 1.0) ;
  vector_init (blvec1, 1.0) ;


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
     resultatf2=mncblas_snrm2_static(VECSIZE, blvec1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);

  printf ("mncblas_snrm2_openmp nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatf2);
  printf ("resultat en flops : %f\n",m_Flops) ;
  printf("\n");


  if(resultatf = resultatf2){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



  printf("======================================================================\n\n");

//============================================================================================================================//


//====================================================vecteur double===========================================================//
  vector_init_double(vecd1,2.0);
  vector_init_double(blvecd1,2.0);

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

  if(resultatd2 == resultatd2){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



  printf("======================================================================\n\n");


//============================================================================================================================//

//====================================================vecteur complex_simple===========================================================//
  struct complex_simple x;
  x.real = 2.0;
  x.imaginary = 3.0;
  vector_init_csimple(veccs1,x);
  vector_init_csimple(blveccs1,x);

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

  if(resultatcs2 == resultatcs){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }



  printf("=============================================================================\n\n");


//=====================================================================================================================================//


//====================================================vecteur complex_double===========================================================//
  struct complex_double y;
  y.real = 4.0;
  y.imaginary = 5.0;
  vector_init_cdouble(veccd1,y);
  vector_init_cdouble(blveccd1,y);

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


  if(resultatcd == resultatcd2){
    printf ("Résultats entre mncblas et mnblas_openmp identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre mncblas et mnblas_openmp différents\n") ;
  }


  printf("=============================================================================\n\n");
//=====================================================================================================================================//


}
