/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2005, University of California
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
 *
 ********************************************************************************/


#ifndef __MATRIX_H__
#define __MATRIX_H__ 

#include <stdio.h>

typedef enum FIND_OP {LT=101, LEQ=102, EQ=103, GEQ=104, GT=105, NE=106} FIND_OP;
typedef enum PRINT_PREC {HUMAN=1001, MACHINE=1002} PRINT_PREC;

typedef struct rect {
	unsigned int d;
	double **boundary;
	FIND_OP *opl;
	FIND_OP *opr;
} Rect;
	
Rect* new_rect(unsigned int d);
Rect* new_dup_rect(Rect* oldR);
Rect* new_drect(double **drect, int d);
void delete_rect(Rect* rect);
unsigned int matrix_constrained(int *p, double **X, unsigned int n1, unsigned int n2, 
				Rect *rect);
void print_rect(Rect *r, FILE* outfile);
double rect_area(Rect* rect);
double rect_area_maxd(Rect* rect, unsigned int maxd);
void rect_unnorm(Rect* r, double **rect, double normscale);
double **get_data_rect(double **X, unsigned int N, unsigned int d);

void normalize(double **Xall, double **rect, int N, int d, double normscale);
void zero(double **M, unsigned int n1, unsigned int n2);
int isZero(double **M, unsigned int m, int sym);
void id(double **M, unsigned int n);
double ** new_id_matrix(unsigned int n);
double ** new_zero_matrix(unsigned int n1, unsigned int n2);
int ** new_zero_imatrix(unsigned int n1, unsigned int n2);
double ** new_matrix(unsigned int m, unsigned int n);
double ** new_matrix_bones(double *v, unsigned int n1, unsigned int n2);
int ** new_imatrix_bones(int *v, unsigned int n1, unsigned int n2);
int ** new_t_imatrix(int** M, unsigned int n1, unsigned int n2);
int ** new_imatrix(unsigned int n1, unsigned int n2);
double ** new_t_matrix(double** M, unsigned int n1_old, unsigned int n2_old);
double ** new_dup_matrix(double** M, unsigned int n1, unsigned int n2);
int ** new_dup_imatrix(int** M, unsigned int n1, unsigned int n2);
double ** new_shift_matrix(double** M, unsigned int n1, unsigned int n2);
void dup_matrix(double** M1, double **M2, unsigned int n1, unsigned int n2);
void dup_imatrix(int** M1, int **M2, unsigned int n1, unsigned int n2);
void swap_matrix(double **M1, double **M2, unsigned int n1, unsigned int n2);
double ** new_bigger_matrix(double** M, unsigned int n1, unsigned int n2, 
			    unsigned int n1_new, unsigned int n2_new);
int ** new_bigger_imatrix(int** M, unsigned int n1, unsigned int n2, 
			  unsigned int n1_new, unsigned int n2_new);
double ** new_normd_matrix(double** M, unsigned int n1, unsigned int n2, 
		double **rect, double normscale);
void delete_matrix(double** m);
void delete_imatrix(int** m);

void check_means(double *mean, double *q1, double *median, double *q2, unsigned int n);
void center_columns(double **M, double *center, unsigned int n1, unsigned int n2);
void center_rows(double **M, double *center, unsigned int n1, unsigned int n2);
void norm_columns(double **M, double *norm, unsigned int n1, unsigned int n2);
void sum_of_columns_f(double *s, double **M, unsigned int n1, unsigned int n2,
		      double(*f)(double));
void sum_of_columns(double *s, double **M, unsigned int n1, unsigned int n2);
void sum_of_each_column_f(double *s, double **M, unsigned int *n1, 
			  unsigned int n2, double(*f)(double));
void wmean_of_columns(double *mean, double **M, unsigned int n1, unsigned int n2, 
		      double *weight);
void wvar_of_columns(double *var, double **M, unsigned int n1, unsigned int n2,
		     double *weight);
void wmean_of_columns_f(double *mean, double **M, unsigned int n1, unsigned int n2, 
			double *weight, double(*f)(double));
void wmean_of_rows(double *mean, double **M, unsigned int n1, unsigned int n2, 
		   double *weight);
void wmean_of_rows_f(double *mean, double **M, unsigned int n1, unsigned int n2, 
		     double *weight, double(*f)(double));
void wcov_of_columns(double **cov, double **M, double *mean, unsigned int n1, 
		     unsigned int n2, double *weight);
void wcovx_of_columns(double **cov, double **M1, double **M2, double *mean1, double *mean2, 
		      unsigned int T,  unsigned int n1, unsigned int n2, double *weight);

void add_matrix(double a, double **M1, double b, double **M2, unsigned int n1, 
		unsigned int n2);
double **new_p_submatrix(int *p, double **v, unsigned int nrows, unsigned int ncols,
			 unsigned int col_offset);
void sub_p_matrix(double **V, int *p, double **v, unsigned int nrows, 
		  unsigned int lenp, unsigned int col_offset);
double **new_p_submatrix_rows(int *p, double **v, unsigned int nrows, 
			      unsigned int ncols, unsigned int row_offset);
void sub_p_matrix_rows(double **V, int *p, double **v, unsigned int ncols, 
		       unsigned int lenp, unsigned int row_offset);
void copy_p_matrix(double **V, int *p1, int *p2, double **v, unsigned int n1, 
		   unsigned int n2);
void add_p_matrix(double a, double **V, int *p1, int *p2, double b, double **v, 
		  unsigned int n1, unsigned int n2);

double* ones(unsigned int n, double scale);
double* dseq(double from, double to, double by);
int* iseq(double from, double to);
 
int* find(double *V, unsigned int n, FIND_OP op, double val, unsigned int* len);
int* find_col(double **V, int *p, unsigned int n, unsigned int var, 
	      FIND_OP op, double val, unsigned int* len);

double kth_smallest(double a[], int n, int k);
double quick_select(double arr[], int n, int k);
void quantiles_of_columns(double **Q, double *q, unsigned int m, double **M, 
			  unsigned int n1, unsigned int n2, double *w);
void quantiles(double *qs, double *q, unsigned int m, double *v,
	       double *w, unsigned int n);

void printMatrix(double **M, unsigned int n, unsigned int col, FILE *outfile);
void printIMatrix(int **matrix, unsigned int n, unsigned int col, FILE *outfile);
void printMatrixT(double **M, unsigned int n, unsigned int col, FILE *outfile);
void mean_to_file(const char *file_str, double **M, unsigned int T, unsigned int n);
void vector_to_file(const char* file_str, double *quantiles, unsigned int n);
void matrix_to_file(const char* file_str, double** matrix, unsigned int n1, unsigned int n2);
void intmatrix_to_file(const char* file_str, int** matrix, unsigned int n1, unsigned int n2);
void matrix_t_to_file(const char* file_str, double** matrix, unsigned int n1, unsigned int n2);
void printVector(double *v, unsigned int n, FILE *outfile, PRINT_PREC type);
void printSymmMatrixVector(double **m, unsigned int n, FILE *outfile, 
			   PRINT_PREC type);
void ivector_to_file(const char* file_str, int *vector, unsigned int n);
void uivector_to_file(const char *file_str, unsigned int *iv, unsigned int n);

double* new_dup_vector(double* vold, unsigned int n);
double* new_zero_vector(unsigned int n);
double* new_vector(unsigned int n);
void dupv(double *v, double* vold, unsigned int n);
void dup_col(double **M, unsigned int col, double *v, unsigned int n);
void swap_vector(double **v1, double **v2);
void zerov(double*v, unsigned int n);
void add_vector(double a, double *v1, double b, double *v2, unsigned int n);
void add_p_vector(double a, double *V, int *p, double b, double *v, unsigned int n);
void copy_p_vector(double *V, int *p, double *v, unsigned int n);
void copy_sub_vector(double *V, int *p, double *v, unsigned int n);
double* new_sub_vector(int *p, double *v, unsigned int n);
void scalev(double *v, unsigned int n, double scale);
void scalev2(double *v, unsigned int n, double *scale);
void centerv(double *v, unsigned int n, double scale);
void normv(double *v, unsigned int n, double* norm);
double sum_fv(double *v, unsigned int n, double(*f)(double));
double sumv(double *v, unsigned int n);
double meanv(double *v, unsigned int n);
int equalv(double *v1, double *v2, int n);

int* new_ivector(unsigned int n);
int* new_dup_ivector(int *iv, unsigned int n);
void dupiv(int *iv_new, int *iv, unsigned int n);
void zeroiv(int*v, unsigned int n);
void swap_ivector(int **v1, int **v2);
int *new_ones_ivector(unsigned int n, int scale);
int *new_zero_ivector(unsigned int n);
void iones(int *iv, unsigned int n, int scale);
void printIVector(int *iv, unsigned int n, FILE *outfile);
void copy_p_ivector(int *V, int *p, int *v, unsigned int n);
void copy_sub_ivector(int *V, int *p, int *v, unsigned int n);
int* new_sub_ivector(int *p, int *v, unsigned int n);
int sumiv(int *v, unsigned int n);
int meaniv(int *iv, unsigned int n);
void add_ivector(int *v1, int *v2, unsigned int n);

unsigned int* new_uivector(unsigned int n);
unsigned int* new_dup_uivector(unsigned int *iv, unsigned int n);
void dupuiv(unsigned int *iv_new, unsigned int *iv, unsigned int n);
void zerouiv(unsigned int *v, unsigned int n);
unsigned int *new_ones_uivector(unsigned int n, unsigned int scale);
unsigned int *new_zero_uivector(unsigned int n);
void uiones(unsigned int *iv, unsigned int n, unsigned int scale);
void printUIVector(unsigned int *iv, unsigned int n, FILE *outfile);
void copy_p_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n);
void copy_sub_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n);
unsigned int* new_sub_uivector(int *p, unsigned int *v, unsigned int n);
unsigned int sumuiv(unsigned int *v, unsigned int n);
unsigned int meanuiv(unsigned int *iv, unsigned int n);

double max(double *v, unsigned int n, unsigned int *which);
double min(double *v, unsigned int n, unsigned int *which);
double sq(double x);
double myfmax(double a, double b);
double myfmin(double a, double b);

double vmult(double *v1, double *v2, int n);


#endif
