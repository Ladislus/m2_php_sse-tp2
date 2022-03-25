#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <cstring>

#define SSE
#define BLOCK_SIZE 8
#define ELEM_SIZE 4

void print_mat(float *A, size_t dim) {
    printf("Matrice : \n");
    for (size_t i = 0; i < dim; ++i) {
//        printf("[ ");
        for (size_t j = 0; j < dim; ++j) {
            printf("%f  ", A[i * dim + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void mul_drip_naive(float *A, float *B, float *res, size_t dim) {
    /*
     * Truc cool sur l'auto-save
     */
    printf("A l'appel de mul_drip_naive : \n");
    print_mat(res, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                res[i * dim + k] += A[i * dim + j] * B[j * dim + k];
                print_mat(res, dim);
            }
        }
    }
}

void mul_drip_sse(float *A, float *B, float *res, size_t dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            __m128 rA = _mm_load1_ps(A + i * dim + j);
            for (int k = 0; k < dim; k += ELEM_SIZE) {
                __m128 rB = _mm_load_ps(B + j * dim + k);
                rB = _mm_mul_ps(rA, rB);
                __m128 r_res = _mm_load_ps(res + i * dim + k);
                r_res = _mm_add_ps(r_res, rB);
                _mm_store_ps(res + (i * dim + k), r_res);
            }
        }
    }
}

void naive_mul(float *A, float *B, float *res, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            int index = i * dim + j;
            float tmp_res = 0.f;
            for (size_t k = 0; k < dim; ++k) {
                tmp_res += A[i * dim + k] * B[k * dim + j];
            }
            res[index] = tmp_res;
        }
    }
}

void check(float *A, float *B, float *res, size_t dim) {
    float *exp = (float *) malloc(dim * dim * sizeof(float));

    naive_mul(A, B, exp, dim);
    printf("la bonne : \n");
    print_mat(exp, dim);

    for (size_t i = 0; i < dim * dim; i++) {
        if (exp[i] != res[i]) {
            printf("Value at %lu differs: %f\n", i, exp[i] - res[i]);
        }
    }
}

int main() {
    size_t dim = 1 * BLOCK_SIZE;

    srand(time(nullptr));

#ifdef SSE
    float * A = (float*) _mm_malloc(dim*dim*sizeof(float), 16);
    float * B = (float*) _mm_malloc(dim*dim*sizeof(float), 16);
    float * C = (float*) _mm_malloc(dim*dim*sizeof(float), 16);

    memset(C, 0, dim*dim*sizeof(float));

    for (int i = 0; i < dim*dim; ++i) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
    }
    mul_drip_sse(A, B, C, dim);
    printf("La notre \n");
    print_mat(C, dim);

#else
    float *A = (float *) malloc(dim * dim * sizeof(float));
    float *B = (float *) malloc(dim * dim * sizeof(float));
    float *C = (float *) calloc(dim * dim, sizeof(float));

    printf("Après calloc : \n");
    print_mat(C, dim);

    for (size_t i = 0; i < dim * dim; i++) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
    }
    print_mat(A, dim);
    print_mat(B, dim);

    // your code or function here
//    naive_mul(A, B, C, dim);
//    print_mat(C, dim);

    mul_drip_naive(A, B, C, dim);
    print_mat(C, dim);


// you can activate check by adding -DCHECK_MUL to your command line
#endif
#ifdef CHECK_MUL
    check(A, B, C, dim);
#endif

    free(A);
    free(B);
    free(C);

    return 0;
}
