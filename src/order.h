#ifndef __ORDER_H__
#define __ORDER_H__

int compareRank(const void* a, const void* b);
int* order(double *s, unsigned int n);
int* rank(double *s, unsigned int n);
unsigned int* rand_indices(unsigned int N);

#endif

