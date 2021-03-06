
typedef enum {MNCblasRowMajor=101, MNCblasColMajor=102} MNCBLAS_LAYOUT;
typedef enum {MNCblasNoTrans=111, MNCblasTrans=112, MNCblasConjTrans=113} MNCBLAS_TRANSPOSE;
typedef enum {MNCblasUpper=121, MNCblasLower=122} MNCBLAS_UPLO;
typedef enum {MNCblasNonUnit=131, MNCblasUnit=132} MNCBLAS_DIAG;
typedef enum {MNCblasLeft=141, MNCblasRight=142} MNCBLAS_SIDE;

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions
 * ===========================================================================
 */


/*
  BLAS copy
*/


void mncblas_scopy(const int N, const float *X, const int incX,
                 float *Y, const int incY);

void mncblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY);


void mncblas_ccopy(const int N, const void *X, const int incX,
                 void *Y, const int incY);


void mncblas_zcopy(const int N, const void *X, const int incX,
                 void *Y, const int incY);



void mncblas_scopy_static(const int N, const float *X, const int incX,
                 float *Y, const int incY);

void mncblas_dcopy_static(const int N, const double *X, const int incX,
                 double *Y, const int incY);


void mncblas_ccopy_static(const int N, const void *X, const int incX,
                 void *Y, const int incY);


void mncblas_zcopy_static(const int N, const void *X, const int incX,
                 void *Y, const int incY);



/*
  end COPY BLAS
*/

/*
  BLAS SWAP
*/

void mncblas_sswap(const int N, float *X, const int incX,
                 float *Y, const int incY);

void mncblas_dswap(const int N, double *X, const int incX,
                 double *Y, const int incY);

void mncblas_cswap(const int N, void *X, const int incX,
                 void *Y, const int incY);

void mncblas_zswap(const int N, void *X, const int incX,
                 void *Y, const int incY);

/*
  END SWAP
*/


/*

  BLAS DOT

*/

float  mncblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);
double mncblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu);
void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc);

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu);
void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc);

//										Version Concurente								//

float  mncblas_sdot_static(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);
double mncblas_ddot_static(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

void   mncblas_cdotu_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu);
void   mncblas_cdotc_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc);

void   mncblas_zdotu_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu);
void   mncblas_zdotc_sub_static(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc);

/*
  END BLAS DOT
*/



/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */

/* BLAS NRM2 */

float mncblas_snrm2(const int N,const float *X,const int incX);

double mncblas_dnrm2(const int N,const double *X,const int incX);

float mncblas_scnrm2(const int N,const void *X,const int incX);

double  mncblas_dznrm2(const int N,const void *X,const int incX);

//										Version Concurente								//

float mncblas_snrm2_static(const int N,const float *X,const int incX);

double mncblas_dnrm2_static(const int N,const double *X,const int incX);

float mncblas_scnrm2_static(const int N,const void *X,const int incX);

double  mncblas_dznrm2_static(const int N,const void *X,const int incX);

//										Version Vectorisé								//

float mncblas_snrm2_vector(const int N,const float *X,const int incX);

double mncblas_dnrm2_vector(const int N,const double *X,const int incX);

float mncblas_scnrm2_vector(const int N,const void *X,const int incX);

double  mncblas_dznrm2_vector(const int N,const void *X,const int incX);

 /* END BLAS NRM2 */

/* BLAS ASUM */
float mncblas_sasum(const int N, const float *X, const int incX);

double mncblas_dasum(const int N, const double *X, const int incX);

float mncblas_scasum(const int N, const void *X, const int incX);

double mncblas_dzasum(const int N, const void *X, const int incX);

//										Version Concurente								//

float mncblas_sasum_static(const int N, const float *X, const int incX);

double mncblas_dasum_static(const int N, const double *X, const int incX);

float mncblas_scasum_static(const int N, const void *X, const int incX);

double mncblas_dzasum_static(const int N, const void *X, const int incX);

 /* END BLAS ASUM */

/* BLAS AXPY */

void mncblas_saxpy(const int N, const float alpha, const float *X,const int incX, float *Y, const int incY);

void mncblas_daxpy(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY);

void mncblas_caxpy(const int N, const void *alpha, const void *X,const int incX, void *Y, const int incY);

void mncblas_zaxpy(const int N, const void *alpha, const void *X,const int incX, void *Y, const int incY);

//										Version Concurente								//

void mncblas_saxpy_static(const int N, const float alpha, const float *X,const int incX, float *Y, const int incY);

void mncblas_daxpy_static(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY);

void mncblas_caxpy_static(const int N, const void *alpha, const void *X,const int incX, void *Y, const int incY);

void mncblas_zaxpy_static(const int N, const void *alpha, const void *X,const int incX, void *Y, const int incY);

//										Version Vectorisé								//

void mncblas_saxpy_vector(const int N, const float alpha, const float *X,const int incX, float *Y, const int incY);

void mncblas_daxpy_vector(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY);

void mncblas_caxpy_vector(const int N, const void *alpha, const void *X,const int incX, void *Y, const int incY);

void mncblas_zaxpy_vector(const int N, const void *alpha, const void *X,const int incX, void *Y, const int incY);

/*END BLAS AXPY*/

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */



void mncblas_sgemv(const MNCBLAS_LAYOUT layout,
                 const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);

void mncblas_dgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);

void mncblas_cgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY);

void mncblas_zgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY);

void mncblas_sgemv_static(const MNCBLAS_LAYOUT layout,
                 const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);

void mncblas_dgemv_static(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);

void mncblas_cgemv_static(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY);

void mncblas_zgemv_static(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY);



/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */



void mncblas_sgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void mncblas_sgemm_static(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);


void mncblas_dgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

void mncblas_dgemm_static(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);


void mncblas_cgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);

void mncblas_cgemm_static(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);

void mncblas_zgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);

void mncblas_zgemm_static(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);



