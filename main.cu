#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

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
                          const uint64_t num_elements
                         ) {

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ((idx & 1) == 0) return;

  uint64_t num1 = idx + left_border;
  uint64_t num2 = num1 + 2;

  if (num2 <= num_elements) {
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
}

bool is_prime(uint32_t number) {
  if (number % 2 == 0) return false;
  for (uint32_t i = 3; i <= (uint32_t)sqrt(number); i+=2) {
  if (number % i == 0) return false;
  }
  return true;
}

void prime_dividers(uint32_t last_numer, uint32_t *primes, uint32_t *primes_size) {
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

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <start_number> <end_number>\n", argv[0]);
    return 1;
  }
  uint64_t left_border = MAX(llabs(atoll(argv[1])), 3);
  uint64_t right_border = llabs(atoll(argv[2]));
  if (right_border <= left_border) {
    printf("Last number must be greater than the first number.\n");
    return 1;
  }

  sscanf(argv[1], "%llu", &left_border);
  sscanf(argv[2], "%llu", &right_border);
  if ((left_border & 1) != 0) --left_border;
  left_border = MAX(2, left_border);
  uint64_t borders_size = right_border - left_border;

  uint32_t h_prime_divs[(uint32_t)sqrt(right_border)];

  uint32_t primes_divs_size = 0;
  prime_dividers(right_border, h_prime_divs, &primes_divs_size);

  uint32_t *d_prime_divs = NULL;
  gpuErrchk(cudaMalloc(&d_prime_divs, primes_divs_size * sizeof(uint32_t)));

  gpuErrchk(cudaMemcpy(d_prime_divs, h_prime_divs, primes_divs_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

  uint32_t results_num = borders_size;
  uint32_t results_size = sizeof(uint32_t) * results_num;

  uint64_t *d_results;
  gpuErrchk(cudaMalloc(&d_results, results_size));

  uint32_t *d_result_count;
  gpuErrchk(cudaMalloc(&d_result_count, sizeof(uint32_t)));
  gpuErrchk(cudaMemset(d_result_count, 0, sizeof(uint32_t)));


  int threadsPerBlock = 128;
  int blocksPerGrid = (borders_size + threadsPerBlock - 1) / threadsPerBlock;
  // printf("\n[Launching kernel with %d blocks and %d threads per block]\n", blocksPerGrid, threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_prime_divs,
                                                primes_divs_size,
                                                d_results,
                                                d_result_count,
                                                left_border,
                                                right_border);

  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaDeviceSynchronize());

  uint32_t h_result_count;
  cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if (h_result_count > 0) {
    uint64_t* h_results = (uint64_t *)malloc(h_result_count * sizeof(uint64_t));
    if (h_results == NULL) {
      fprintf(stderr, "[ERROR] Failed to allocate h_results!\n");
      exit(EXIT_FAILURE);
    }
    cudaMemcpy(h_results, d_results, h_result_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // printf("\nResults:\n");
    // for (int i = 0; i < h_result_count; ++i) {
    //   printf("%u %u\n", h_results[i], h_results[i] + 2);
    // }
    free(h_results);
  }

  printf("\n[Found %u results]\n", h_result_count);

  gpuErrchk(cudaFree(d_results));
  gpuErrchk(cudaFree(d_result_count));

  return 0;
}
