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

vfloat vec1,blvec1;
float resultatf,resultatcs;
vdouble vecd1,blvecd1;
double resultatd,resultatcd;
vcsimple veccs1,blveccs1;
vcdouble veccd1,blveccd1;
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
     resultatf=cblas_sasum (VECSIZE, vec1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("cblas_sasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatf);
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatf=mncblas_sasum (VECSIZE, blvec1, 1) ;
  end = _rdtsc () ;
	m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);


  printf ("mncblas_sasum: nombre de cycles: %Ld \n", end-start-residu) ;
  printf ("resultat: %f \n",resultatf) ;
	printf ("resultat en Gflop : %f \n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatf=cblas_sasum (VECSIZE, vec1, 1) ;
  end = _rdtsc () ;
	m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);

  printf ("cblas_sasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatf);
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurFloat(vec1,VECSIZE,blvec1,VECSIZE)){
    printf ("Résultats entre cblas et mnblas identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre cblas et mnblas différents\n") ;
  }



  printf("======================================================================\n\n");

//============================================================================================================================//


//====================================================vecteur double===========================================================//
  vector_init_double(vecd1,2.0);
  vector_init_double(blvecd1,2.0);


printf("=========================VECTEUR Double================================\n");

  start = _rdtsc () ;
     resultatd = cblas_dasum (VECSIZE, vecd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("cblas_dasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatd);
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");


  start = _rdtsc () ;
     resultatd = mncblas_dasum (VECSIZE, blvecd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatd);
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatd = cblas_dasum (VECSIZE, vecd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("cblas_dasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatd);
	printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");


  if(comparaisonVecteurDouble(vecd1,VECSIZE,blvecd1,VECSIZE)){
    printf ("Résultats entre cblas et mnblas identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre cblas et mnblas différents\n") ;
  }



  printf("======================================================================\n\n");

  /*printf("Vector 2 double\n");
  vector_print_double(vecd2);
  vector_print_double(vecd1);*/
//============================================================================================================================//

//====================================================vecteur complex_simple===========================================================//
  struct complex_simple x;
  x.real = 2.0;
  x.imaginary = 3.0;
  vector_init_csimple(veccs1,x);
  vector_init_csimple(blveccs1,x);

printf("=========================VECTEUR COMPLEXES SIMPLES================================\n");

  start = _rdtsc () ;
     resultatcs = cblas_scasum (VECSIZE, veccs1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_scasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcs);
  printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatcs = mncblas_scasum (VECSIZE, veccs1, 1) ;
  end = _rdtsc () ;
  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_scasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcs);
  printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatcs = cblas_scasum (VECSIZE, veccs1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_scasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcs);
  printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurCS(veccs1,VECSIZE,blveccs1,VECSIZE)){
    printf ("Résultats entre cblas et mnblas identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre cblas et mnblas différents\n") ;
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
     resultatcd = cblas_dzasum (VECSIZE, veccd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dzasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcd);
  printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatcd = mncblas_dzasum (VECSIZE, veccd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dzasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcd);
  printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  start = _rdtsc () ;
     resultatcd = cblas_dzasum (VECSIZE, veccd1, 1) ;
  end = _rdtsc () ;

  m_Flops=FLOPS(1,3.4,2*VECSIZE,end-start-residu);
  printf ("mncblas_dzasum nombre de cycles cblas: %Ld \n", end-start-residu) ;
  printf ("resultat: %f\n",resultatcd);
  printf ("resultat en Gflops : %f\n",m_Flops) ;
  printf("\n");

  if(comparaisonVecteurCD(veccd1,VECSIZE,blveccd1,VECSIZE)){
    printf ("Résultats entre cblas et mnblas identiques\n") ;
  }
  else{
    printf ("Erreurs ! Résultats entre cblas et mnblas différents\n") ;
  }


  printf("=============================================================================\n\n");
//=====================================================================================================================================//


}
