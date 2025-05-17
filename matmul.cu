#include "helper.cuh"
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 64;
constexpr int EXPAND_FACTOR = 4;

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE / EXPAND_FACTOR]; // (16, 4)
  __shared__ float Bs[TILE_SIZE / EXPAND_FACTOR][TILE_SIZE]; // (4, 16)

  size_t block_row = blockIdx.y * TILE_SIZE;
  size_t block_col = blockIdx.x * TILE_SIZE;

  size_t thread_row = threadIdx.y * EXPAND_FACTOR;
  size_t thread_col = threadIdx.x * EXPAND_FACTOR;

  float sums[EXPAND_FACTOR][EXPAND_FACTOR] = {0.0f};

  // split along the common dimension
  for (int tile_start = 0; tile_start < N;
       tile_start += TILE_SIZE / EXPAND_FACTOR) {
#pragma unroll
    for (int i = 0; i < EXPAND_FACTOR; ++i) {
      int load_row = block_row + thread_row + i;
      int load_col = tile_start + threadIdx.x;
      if (load_row < M && load_col < N) {
        As[thread_row + i][threadIdx.x] = A[load_row * N + load_col];
      } else {
        As[thread_row + i][threadIdx.x] = 0.0f;
      }
    }

    int load_row = tile_start + threadIdx.y;
    if (block_col + thread_col + (EXPAND_FACTOR - 1) < K) {
      if (load_row < N) {
        float4 float4_b = reinterpret_cast<const float4 *>(&B[load_row * K + block_col + thread_col])[0];
        Bs[threadIdx.y][thread_col + 0] = float4_b.x;
        Bs[threadIdx.y][thread_col + 1] = float4_b.y;
        Bs[threadIdx.y][thread_col + 2] = float4_b.z;
        Bs[threadIdx.y][thread_col + 3] = float4_b.w;
      } else {
        Bs[threadIdx.y][thread_col + 0] = 0.0f;
        Bs[threadIdx.y][thread_col + 1] = 0.0f;
        Bs[threadIdx.y][thread_col + 2] = 0.0f;
        Bs[threadIdx.y][thread_col + 3] = 0.0f;
      }
    } else {
#pragma unroll
      for (int j = 0; j < EXPAND_FACTOR; ++j) {
        int load_col = block_col + thread_col + j;
        if (load_row < N && load_col < K) {
          Bs[threadIdx.y][thread_col + j] = B[load_row * K + load_col];
        } else {
          Bs[threadIdx.y][thread_col + j] = 0.0f;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < (TILE_SIZE / EXPAND_FACTOR); ++k) {
      float a_frag[EXPAND_FACTOR];
#pragma unroll
      for (int i = 0; i < EXPAND_FACTOR; ++i) {
        a_frag[i] = As[thread_row + i][k];
      }

      float b_frag[EXPAND_FACTOR];
      float4 float4_b_frag_val = reinterpret_cast<const float4 *>(&Bs[k][thread_col])[0];
      b_frag[0] = float4_b_frag_val.x;
      b_frag[1] = float4_b_frag_val.y;
      b_frag[2] = float4_b_frag_val.z;
      b_frag[3] = float4_b_frag_val.w;

#pragma unroll
      for (int i = 0; i < EXPAND_FACTOR; ++i) {
#pragma unroll
        for (int j = 0; j < EXPAND_FACTOR; ++j) {
          sums[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    // 在所有线程都算好之前，任何线程都不准偷跑，不然如果某个线程提前进入下一轮循环，
    // 会覆盖shm上的数据
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < EXPAND_FACTOR; ++i) {
#pragma unroll
    for (int j = 0; j < EXPAND_FACTOR; ++j) {
      int global_row = block_row + thread_row + i;
      int global_col = block_col + thread_col + j;
      if (global_row < M && global_col < K) {
        C[global_row * K + global_col] = sums[i][j];
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int M, int N, int K) {
  static_assert(EXPAND_FACTOR % 4 == 0, "EXPAND_FACTOR must be divisible by 4");
  static_assert(TILE_SIZE >= EXPAND_FACTOR,
                "TILE_SIZE must be greater than or equal to EXPAND_FACTOR");
  static_assert(TILE_SIZE % EXPAND_FACTOR == 0,
                "TILE_SIZE must be divisible by EXPAND_FACTOR");
  dim3 threadsPerBlock(TILE_SIZE / EXPAND_FACTOR, TILE_SIZE / EXPAND_FACTOR);
  dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

  cudaMemset(C, 0, sizeof(float) * M * K);
  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}

int main() {
  // test the solve function
  const int M = 2, N = 2, K = 2;
  float A[M * N] = {0.0f};
  float B[N * K] = {0.0f};
  for (int i = 0; i < M * N; ++i) {
    A[i] = (i + 1) * 1.f;
  }
  for (int j = 0; j < N * K; ++j) {
    B[j] = (j + 5)* 1.f;
  }
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
