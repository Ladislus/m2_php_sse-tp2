#include <iostream>
#include <ctime>
#include <cstring>
#include <immintrin.h>

#define SSE
#define BONUS
#define BLOCK_SIZE 8
#define ELEM_SIZE 4

void print_matrix(const float *const A, const size_t &dim, const std::string &name) {
	std::clog << name << ": " << std::endl;
	for (size_t i = 0; i < dim; ++i) {
		std::clog << "[ ";
		for (size_t j = 0; j < dim; ++j) std::clog << A[i * dim + j] << " ";
		std::clog << "]" << std::endl;
	}
}

void naive(const float *const A, const float *const B, float *const res, const size_t &dim) {
	for (size_t i = 0; i < dim; ++i)
		for (size_t j = 0; j < dim; ++j) {
			float tmp_res = 0.f;
			for (size_t k = 0; k < dim; ++k) tmp_res += A[i * dim + k] * B[k * dim + j];
			res[i * dim + j] = tmp_res;
		}
}

void naive_drip(const float *const A, const float *const B, float *const res, const size_t &dim) {
	// Spent 1h debugging this.
	// We had the right answer, we just forgot to save the code
	for (size_t i = 0; i < dim; ++i)
		for (size_t j = 0; j < dim; ++j)
			for (size_t k = 0; k < dim; ++k)
				res[i * dim + k] += A[i * dim + j] * B[j * dim + k];
}

void sse(const float *const A, const float *const B, float *const res, const size_t &dim) {
	for (size_t i = 0; i < dim; ++i)
		for (size_t j = 0; j < dim; ++j) {
			__m128 rA = _mm_load1_ps(A + i * dim + j);
			for (size_t k = 0; k < dim; k += ELEM_SIZE) {
				__m128 rB = _mm_load_ps(B + j * dim + k);
				rB = _mm_mul_ps(rA, rB);
				__m128 rRes = _mm_load_ps(res + i * dim + k);
				rRes = _mm_add_ps(rRes, rB);
				_mm_store_ps(res + (i * dim + k), rRes);
			}
		}
}

//void sseDrip(const float *const A, const float *const B, float *const res, const size_t &dim) {}

void check(const float *const A, const float *const B, const float *const res, const size_t &dim) {
	auto *const exp = (float *) malloc(dim * dim * sizeof(float));

	naive(A, B, exp, dim);
	print_matrix(exp, dim, "Expected");

	for (size_t i = 0; i < (dim * dim); i++)
		if (exp[i] != res[i])
			std::cerr << "Value at " << i << " differs: " << exp[i] - res[i] << std::endl;

	free(exp);
}

int main() {
	const size_t dim = 1 * BLOCK_SIZE;
	const size_t size = dim * dim;

	srand(time(nullptr));

#ifdef SSE
	auto *const A = (float *) _mm_malloc(size * sizeof(float), 16);
	auto *const B = (float *) _mm_malloc(size * sizeof(float), 16);
	auto *const C = (float *) _mm_malloc(size * sizeof(float), 16);

	memset(C, 0, size * sizeof(float));

	for (size_t i = 0; i < size; ++i) {
		A[i] = rand() % 5;
		B[i] = rand() % 5;
	}

	sse(A, B, C, dim);
	print_matrix(C, dim, "Restult");

#elif defined(BONUS)

	auto *const A = (float*) _mm_malloc(size * sizeof(float), 16);
	auto *const B = (float*) _mm_malloc(size * sizeof(float), 16);
	auto *const C = (float*) _mm_malloc(size * sizeof(float), 16);
	memset(C, 0, size * sizeof(float));

	for (size_t i = 0; i < size; ++i) {
		A[i] = rand() % 5;
		B[i] = rand() % 5;
	}

#else
	auto *const A = (float*) malloc(size * sizeof(float));
	auto *const B = (float*) malloc(size * sizeof(float));
	auto *const C = (float*) calloc(size, sizeof(float));

	for (size_t i = 0; i < dim * dim; i++) {
		A[i] = rand() % 5;
		B[i] = rand() % 5;
	}

	print_matrix(A, dim, "A");
	print_matrix(B, dim, "B");

	naive_drip(A, B, C, dim);
	print_matrix(C, dim, "C");
#endif

#ifdef CHECK_MUL
	check(A, B, C, dim);
#endif

	free(A);
	free(B);
	free(C);

	return 0;
}
