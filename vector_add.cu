#include "helper.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  const int N = 1024;
  const int size = N * sizeof(float);
  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C = new float[N];
  // Initialize input vectors
  for (int i = 0; i < N; i++) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i);
  }
  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc((void **)&d_A, size));
  CHECK_CUDA(cudaMalloc((void **)&d_B, size));
  CHECK_CUDA(cudaMalloc((void **)&d_C, size));
  CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
  CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    std::cout << h_C[i] << " ";
  }
  std::cout << "\n";
  // Clean up
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  return 0;
}
