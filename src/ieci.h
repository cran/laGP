#ifndef __IECI_H__
#define __IECI_H__

typedef enum STYPE {UL=3001, MEAN=3002, NORM=3004} Stype;

double EI(const double m, const double s2, const int df, const double fmin);
void calc_ecis(const int m, double *ktKik, double *s2p, const double phi, 
	       const double g, double *badj, double *tm, const double tdf, 
	       const double fmin);
double calc_ieci(const int m, double *ktKik, double *s2p, const double phi, 
		 const double g, double *badj, double *tm, const double tdf, 
		 const double fmin, double *w);
double calc_alc(const int m, double *ktKik, double *s2p, const double phi, 
		double *badj, const double tdf, double *w);
void calc_ktKikx(double *ktKik, const int m, double **k, const int n,
		 double *g, const double mui, double *kxy, double **Gmui_util,
		 double *ktGmui_util, double *ktKikx);
void rbetter_R(int *n_in, int *m_in, double *rect_in, double *ystar_in, 
	       double *X_out);
void MC_al_eiey(const unsigned int nc, const unsigned int nn, double *mu, 
	double *s, const double fnorm, double **cmu, double **cs, double *cnorms, 
	double *lambda, const double alpha, const double ymin, double *equal,
	const unsigned int N, double *eys, double *eis);
void MC_alslack_eiey(const unsigned int nc, const unsigned int nn, double *mu, 
	double *s, const double fnorm, double **cmu, double **cs, double *cnorms, 
	double *lambda, const double alpha, double fmin, double *equal, 
	const unsigned int N, double *eys, double *eis);
void calc_alslack_eiey(const unsigned int nc, const unsigned int nn, 
	double *mu, double *s, const double fnorm, double **cmu, double **cs, 
	double *cnorms, double *lambda, const double alpha, const double ymin,
	double *equal, double *eys, double *eis);
void draw_slacks(const unsigned int nc, const unsigned int nn, 
  double **cmu, double **cs, double *cnorms, double *lambda, 
  const double alpha, double *equal, double **slacks, Stype stype);

#endif

