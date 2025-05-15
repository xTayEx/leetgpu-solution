#include "helper.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas.h>

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  size_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int m = row_idx; m < M; m += blockDim.y * gridDim.y) {
    for (int k = col_idx; k < K; k += blockDim.x * gridDim.x) {
      #pragma unroll
      for (int n = 0; n < N; ++n) {
        C[m * K + k] += A[m * N + n] * B[n * K + k];
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main() {
  // test the solve function
  const int M = 2, N = 2, K = 2;
  float A[M * N] = {1, 2, 3, 4};
  float B[M * N] = {5, 6, 7, 8};
  float C[M * K] = {0};
  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc((void **)&d_A, M * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, N * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * K * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
  solve(d_A, d_B, d_C, M, N, K);
  cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      printf("%f ", C[i * K + j]);
    }
    printf("\n");
  }
}
