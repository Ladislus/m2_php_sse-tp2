#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void print_mat(float* A, size_t dim){
    printf("Matrice : \n");
    for (size_t i = 0; i < dim; ++i) {
        printf("[ ");
        for (size_t j = 0; j < dim; ++j) {
            printf("%f  ", A[i*dim + j]);
        }
        printf("]\n");
    }
    printf("\n");
}

void naive_mul(float *A, float *B, float *res, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            int index = i * dim + j;
            float tmp_res = 0.f;
            for (size_t k = 0; k < dim; ++k) {
                tmp_res += A[i*dim+k] * B[k*dim+j];
            }
            res[index] = tmp_res;
        }
    }
}

void check(float *A, float *B, float *res, size_t dim) {
    float *exp = (float *) malloc(dim * dim * sizeof(float));

    naive_mul(A, B, exp, dim);

    for (size_t i = 0; i < dim * dim; i++) {
        if (exp[i] != res[i]) {
            printf("Value at %lu differs: %f\n", i, exp[i] - res[i]);
        }
    }
}

int main() {
    size_t dim = 2;

    float *A = (float *) malloc(dim * dim * sizeof(float));
    float *B = (float *) malloc(dim * dim * sizeof(float));
    float *C = (float *) malloc(dim * dim * sizeof(float));

    for (size_t i = 0; i < dim * dim; i++) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
    }
    print_mat(A, dim);
    print_mat(B, dim);

    // your code or function here
    naive_mul(A, B, C, dim);

    print_mat(C, dim);


// you can activate check by adding -DCHECK_MUL to your command line
#ifdef CHECK_MUL
    check(A, B, C, dim);
#endif

    free(A);
    free(B);
    free(C);

    return 0;
}
