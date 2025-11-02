#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <cuda_runtime.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// #define CHUNK_SIZE 1'000'000
#define CHUNK_SIZE 100'000'000
// #define CHUNK_SIZE 500'000'000
// #define CHUNK_SIZE 1'000'000'000

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void vectorAdd(const uint32_t *prime_divs,
                          const uint32_t prime_divs_size,
                          uint64_t *results,
                          uint32_t *result_count,
                          const uint64_t left_border,
                          const uint64_t right_border
                         ) {

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t odd_offset = (uint64_t)idx * 2;
  uint64_t num1 = odd_offset + left_border;
  uint64_t num2 = num1 + 2;
  if (num2 > right_border) return;
  bool prime_status = true;

  for (uint32_t i = 0; i < prime_divs_size; ++i) {
    uint32_t p_div = prime_divs[i];
    if (p_div * p_div > num2)
      break;
    if (num1 % p_div == 0 || num2 % p_div == 0) {
      prime_status = false;
      break;
    }
  }
  if (prime_status) {
    uint32_t write_index = atomicAdd(result_count, 1);
    results[write_index] = num1;
  }
}

bool is_prime(uint32_t number) {
  if (number % 2 == 0) return false;
  for (uint32_t i = 3; i <= (uint32_t)sqrt(number); i+=2) {
  if (number % i == 0) return false;
  }
  return true;
}

void prime_dividers(uint64_t last_numer, uint32_t *primes, uint32_t *primes_size) {
  uint32_t p_size = 0;
  primes[p_size++] = 2;
  for (uint32_t i = 3; i <= (uint32_t)(sqrt(last_numer)); i+=2) {
  bool prime_status = is_prime(i);
  if (prime_status) {
    primes[p_size++] = i;
  }
  }
  *primes_size = p_size;
}

void parse_args(int argc, char *argv[], uint64_t *left_border, uint64_t *right_border) {
  if (argc != 3) {
    printf("Usage: %s <start_number> <end_number>\n", argv[0]);
    exit(1);
  }
  char *endptr_left, *endptr_right;
  errno = 0;
  *left_border = llabs(strtoul(argv[1], &endptr_left, 10));
  *right_border = llabs(strtoul(argv[2], &endptr_right, 10));
  if (errno == ERANGE || endptr_left == argv[1] || endptr_right == argv[2]) {
    printf("Worng start or end number\n");
    exit(1);
  }
  *left_border = MAX(*left_border, 3);
  if ((*left_border & 1) == 0) ++*left_border;
  if (*right_border <= *left_border) {
    printf("Last number must be greater than the first number.\n");
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  uint64_t left_border = 0;
  uint64_t right_border = 0;
  parse_args(argc, argv, &left_border, &right_border);

  uint64_t borders_size = right_border - left_border;

  uint64_t chunk_size = MIN(borders_size, CHUNK_SIZE);
  printf("chunk_size %lu\n", chunk_size);
  uint32_t num_chunks = borders_size / chunk_size;
  if (borders_size == chunk_size) num_chunks = 0;
  printf("num_chunks %u\n", num_chunks);

  uint32_t h_prime_divs[(uint32_t)sqrt(right_border)];

  uint32_t h_prime_divs_size = 0;
  prime_dividers(right_border, h_prime_divs, &h_prime_divs_size);

  uint32_t *d_prime_divs = NULL;
  gpuErrchk(cudaMalloc(&d_prime_divs, h_prime_divs_size * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_prime_divs, h_prime_divs, h_prime_divs_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

  uint32_t results_num = chunk_size;
  uint32_t results_size = sizeof(uint64_t) * results_num;

  uint64_t *d_results;
  gpuErrchk(cudaMalloc(&d_results, results_size));

  uint32_t *d_result_count;
  gpuErrchk(cudaMalloc(&d_result_count, sizeof(uint32_t)));


  uint32_t h_results_size = chunk_size;
  uint64_t* h_results = (uint64_t *)malloc(h_results_size * sizeof(uint64_t));
  if (h_results == NULL) {
    fprintf(stderr, "[ERROR] Failed to allocate h_results!\n");
    exit(1);
  }

  uint32_t h_total_result_count = 0;

  for (uint32_t i = 0; i <= num_chunks; ++i) {
    gpuErrchk(cudaMemset(d_result_count, 0, sizeof(uint32_t)));

    uint64_t h_chunk_left_border = left_border + (i * chunk_size);

    uint64_t h_chunk_right_border;
    if (i == num_chunks)
      h_chunk_right_border = right_border;
    else
      h_chunk_right_border = h_chunk_left_border + chunk_size;

    uint32_t threadsPerBlock = 128;
    uint32_t blocksPerGrid = ((chunk_size / 2) + threadsPerBlock - 1) / threadsPerBlock;

    // printf("\n[Launching kernel with %d blocks and %d threads per block]\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_prime_divs,
                                                  h_prime_divs_size,
                                                  d_results,
                                                  d_result_count,
                                                  h_chunk_left_border,
                                                  h_chunk_right_border
                                                 );

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint32_t h_chunk_result_count = 0;

    cudaMemcpy(&h_chunk_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    if (h_chunk_result_count > 0) {
      if (h_results_size < h_chunk_result_count + h_total_result_count) {
        uint64_t new_size = h_total_result_count + h_chunk_result_count;
        uint64_t *h_new_results = (uint64_t *)realloc(h_results, new_size * sizeof(uint64_t));
        if (h_new_results == NULL) {
          fprintf(stderr, "[ERROR] Failed to reallocate h_results!\n");
          free(h_results);
          return 1;
        }
        h_results = h_new_results;
        h_results_size = new_size;
      }
      uint64_t *h_destination_ptr = h_results + h_total_result_count;
      cudaMemcpy(h_destination_ptr, d_results, h_chunk_result_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    h_total_result_count += h_chunk_result_count;

    fprintf(stderr, "Chunk %u [%llu..%llu](%lu)\r", i, h_chunk_left_border, h_chunk_right_border, h_chunk_result_count);
  }

  // printf("\n[Found %u results]\n", h_total_result_count);
  // for (uint32_t i = 0; i < h_total_result_count; ++i) {
  //   printf("%llu..%llu\n", h_results[i], h_results[i] + 2);
  // }

  printf("\n[Found %u results]\n", h_total_result_count);

  free(h_results);
  gpuErrchk(cudaFree(d_prime_divs));
  gpuErrchk(cudaFree(d_results));
  gpuErrchk(cudaFree(d_result_count));

  return 0;
}
