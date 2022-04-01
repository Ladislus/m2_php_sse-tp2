#include <iostream>
#include <ctime>
#include <cstring>
#include <immintrin.h>

#ifdef VERBOSE
	#define LOG(code) code;
#else
	#define LOG(code)
#endif

#define ELEM_SIZE 4

/*
 * Utility function to convert a float register to string (used for printing)
 */
inline std::string register_to_string(const __m128& reg) {
    std::string res;
    res += "| ";
    for(size_t i = 0; i < 4; ++i)
        res += std::to_string((int) reg[i]) + " ";
    res += "|";
    return res;
}

/*
 * Utility function to reduce a float register to a single value
 */
inline float reduce(const __m128& reg) {
    // Unload the register into a float array
	float elems[4];
	_mm_store_ps(elems, reg);

    // Sum the array
	float sum = 0.f;
	for (float elem : elems) sum += elem;

    // Return the sum
	return sum;
}

/*
 * Utility function to print a matrix in a nice format
 * @param A: The pointer to the matrix
 * @param dim: The dimension of the matrix (must be a square matrix of 'dim x dim')
 * @param name: The name printed before the matrix
 */
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
	// Spent 1h debugging this. We had the right answer,
    // we just forgot to save and recompile the code
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

void sse_drip(const float *const A, const float *const B, float *const res, const size_t &dim) {
	// Iterate over A in 4x4 blocks
	for (size_t i = 0; i < dim; i += ELEM_SIZE)
		for (size_t j = 0; j < dim; j += ELEM_SIZE) {
			// Load a block from A
			__m128 rA[4] = {
					_mm_load_ps(A + i * dim + j),
					_mm_load_ps(A + (i + 1) * dim + j),
					_mm_load_ps(A + (i + 2) * dim + j),
					_mm_load_ps(A + (i + 3) * dim + j)
			};
			// Iterate over B by 4x4 blocks on the same row index as A
			for (size_t k = 0; k < dim; k += ELEM_SIZE) {
				// Load the 4x4 block of B
				__m128 rB[4] = {
						_mm_load_ps(B + j * dim + k),
						_mm_load_ps(B + (j + 1) * dim + k),
						_mm_load_ps(B + (j + 2) * dim + k),
						_mm_load_ps(B + (j + 3) * dim + k)
				};

				// Transpose the block (Invert rows & columns)
				_MM_TRANSPOSE4_PS(rB[0], rB[1], rB[2], rB[3]);

				// Multiply each A row fragment by each B column fragment and add to the result
				for (size_t m = 0; m < ELEM_SIZE; ++m)
					for (size_t n = 0; n < ELEM_SIZE; ++n) {
						__m128 r = _mm_mul_ps(rA[m], rB[n]);

                        size_t index = (i + m) * dim + (k + n);
                        float reduction = reduce(r);

                        // Debug log to check if the reduction is correct,
                        // And written to the result matrix in the right place
                        LOG(
                            std::clog << "i: " << i << " j: " << j << " k: " << k << " m: " << m << " n: " << n << " index: " << index << std::endl;
                            std::clog << "A: " << register_to_string(rA[m]) << std::endl << "B: " << register_to_string(rB[n]) << std::endl << "r: " << register_to_string(r) << std::endl;
                        )

						res[index] += reduction;

                        // Debug log to check the current state of the result matrix
                        LOG(
                            print_matrix(res, dim, "Current res");
                            std::clog << "#################################" << std::endl;
                        )
					}
			}
		}
}

void check(const float *const A, const float *const B, const float *const res, const size_t &dim) {
	auto *const exp = (float *) malloc(dim * dim * sizeof(float));

	naive(A, B, exp, dim);
	LOG(print_matrix(exp, dim, "Expected"))

	for (size_t i = 0; i < (dim * dim); i++)
		if (exp[i] != res[i])
			std::cerr << "Value at " << i << " differs: " << exp[i] - res[i] << std::endl;

	free(exp);
}

int main() {
	constexpr size_t dim = 3 * ELEM_SIZE;
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

    LOG(
        print_matrix(A, dim, "A");
        print_matrix(B, dim, "B");
    )
	sse(A, B, C, dim);
	LOG(print_matrix(C, dim, "Restult"))

#elif defined(BONUS)

	auto *const A = (float*) _mm_malloc(size * sizeof(float), 16);
	auto *const B = (float*) _mm_malloc(size * sizeof(float), 16);
	auto *const C = (float*) _mm_malloc(size * sizeof(float), 16);
	memset(C, 0, size * sizeof(float));

	for (size_t i = 0; i < size; ++i) {
		A[i] = rand() % 5;
		B[i] = rand() % 5;
	}

    LOG(
        print_matrix(A, dim, "A");
        print_matrix(B, dim, "B");
    )
	sse_drip(A, B, C, dim);
	LOG(print_matrix(C, dim, "Result");)
#else
	auto *const A = (float*) malloc(size * sizeof(float));
	auto *const B = (float*) malloc(size * sizeof(float));
	auto *const C = (float*) calloc(size, sizeof(float));

	for (size_t i = 0; i < dim * dim; i++) {
		A[i] = rand() % 5;
		B[i] = rand() % 5;
	}

	LOG(
		print_matrix(A, dim, "A");
		print_matrix(B, dim, "B")
	)
	naive_drip(A, B, C, dim);
	LOG(print_matrix(C, dim, "C"))
#endif

#ifdef CHECK_MUL
	check(A, B, C, dim);
#endif

	free(A);
	free(B);
	free(C);

	return EXIT_SUCCESS;
}
