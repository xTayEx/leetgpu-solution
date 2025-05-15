#include "helper.cuh"
#include <cstdio>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 16;

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE * 4];

  size_t row = blockIdx.y * TILE_SIZE;
  size_t col = blockIdx.x * TILE_SIZE * 4;

  size_t ty = threadIdx.y;
  size_t tx = threadIdx.x;

  int m = row + ty;
  int k_start = col + tx * 4;

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;

  for (int n_start = 0; n_start < N; n_start += TILE_SIZE) {
    int a_col = n_start + tx;
    if (m < M && a_col < N) {
      As[ty][tx] = A[m * N + a_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    int b_row = n_start + ty;
    int b_col = k_start;
    if (b_row < N) {
      if (b_col + 3 < K) {
        float4 b_val =
            reinterpret_cast<const float4 *>(&B[b_row * K + b_col])[0];
        Bs[ty][tx * 4] = b_val.x;
        Bs[ty][tx * 4 + 1] = b_val.y;
        Bs[ty][tx * 4 + 2] = b_val.z;
        Bs[ty][tx * 4 + 3] = b_val.w;
      } else {
        for (int i = 0; i < 4; ++i) {
          Bs[ty][tx * 4 + i] =
              (b_col + i < K) ? B[b_row * K + (b_col + i)] : 0.0f;
        }
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        Bs[ty][tx * 4 + i] = 0.0f;
      }
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
      float a = As[ty][i];
      sum0 += a * Bs[i][tx * 4];
      sum1 += a * Bs[i][tx * 4 + 1];
      sum2 += a * Bs[i][tx * 4 + 2];
      sum3 += a * Bs[i][tx * 4 + 3];
    }

    __syncthreads();
  }

  if (m < M) {
    if (k_start < K) {
      C[m * K + k_start] = sum0;
    }
    if (k_start + 1 < K) {
      C[m * K + k_start + 1] = sum1;
    }
    if (k_start + 2 < K) {
      C[m * K + k_start + 2] = sum2;
    }
    if (k_start + 3 < K) {
      C[m * K + k_start + 3] = sum3;
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + TILE_SIZE * 4 - 1) / (TILE_SIZE * 4),
                     (M + TILE_SIZE - 1) / TILE_SIZE);

  cudaMemset(C, 0, sizeof(float) * M * K);
  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
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
