#include "helper.cuh"
#include <boost/program_options.hpp>
#include <boost/program_options/detail/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

namespace po = boost::program_options;

template <int TILE_SIZE = 64, int EXPAND_FACTOR = 4>
__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  __shared__ float As[2][TILE_SIZE][TILE_SIZE / EXPAND_FACTOR]; // (16, 4)
  __shared__ float Bs[2][TILE_SIZE / EXPAND_FACTOR][TILE_SIZE]; // (4, 16)

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
        float4 float4_b = reinterpret_cast<const float4 *>(
            &B[load_row * K + block_col + thread_col])[0];
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
      float4 float4_b_frag_val =
          reinterpret_cast<const float4 *>(&Bs[k][thread_col])[0];
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
    int global_col_start = block_col + thread_col;
    int global_row = block_row + thread_row + i;
    if (global_row >= M) {
      return;
    }
    if (global_col_start + EXPAND_FACTOR - 1 < K) {
#pragma unroll
      for (int j = 0; j < EXPAND_FACTOR / 4; ++j) {
        reinterpret_cast<float4 *>(
            &C[global_row * K + global_col_start + 4 * j])[0] =
            make_float4(sums[i][4 * j + 0], sums[i][4 * j + 1],
                        sums[i][4 * j + 2], sums[i][4 * j + 3]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < EXPAND_FACTOR; ++j) {
        int global_col = global_col_start + j;
        if (global_col >= K) {
          break;
        }
        C[global_row * K + global_col] = sums[i][j];
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int M, int N, int K) {
  constexpr int expand_factor = 4;
  constexpr int tile_size = 64;
  static_assert(expand_factor % 4 == 0, "expand_factor must be divisible by 4");
  static_assert(tile_size >= expand_factor,
                "tile_size must be greater than or equal to expand_factor");
  static_assert(tile_size % expand_factor == 0,
                "tile_size must be divisible by expand_factor");
  dim3 threadsPerBlock(tile_size / expand_factor, tile_size / expand_factor);
  dim3 blocksPerGrid((K + tile_size - 1) / tile_size,
                     (M + tile_size - 1) / tile_size);

  cudaMemset(C, 0, sizeof(float) * M * K);
  matrix_multiplication_kernel<tile_size, expand_factor>
      <<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

bool parse_argument(int &M, int &N, int &K, int argc, const char *const *argv) {
  po::options_description desc("GEMM Options");
  desc.add_options()("m", po::value<int>(), "rows of A")(
      "n", po::value<int>(), "columns of A")("k", po::value<int>(),
                                             "columns of B");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (!vm.count("m") || !vm.count("n") || !vm.count("k")) {
    return false;
  }

  M = vm["m"].as<int>();
  N = vm["n"].as<int>();
  K = vm["k"].as<int>();
  return true;
}

bool read_matrix_from_file(const char *filename, float *matrix,
                           int expected_size) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Failed to open file " << filename << std::endl;
    return false;
  }

  std::string line;
  if (!std::getline(file, line)) {
    std::cerr << "Error: Empty file " << filename << std::endl;
    return false;
  }

  std::istringstream iss(line);
  int count = 0;
  float value;

  while (iss >> value) {
    if (count >= expected_size) {
      std::cerr << "Error: Too many elements in " << filename << " (expected "
                << expected_size << ")" << std::endl;
      return false;
    }
    matrix[count++] = value;
  }

  if (count != expected_size) {
    std::cerr << "Error: Insufficient elements in " << filename << " (expected "
              << expected_size << ", got " << count << ")" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  // test the solve function
  int M, N, K;
  if (!parse_argument(M, N, K, argc, argv)) {
    return 1;
  }
  float *A = new float[M * N];
  float *B = new float[N * K];

  if (!read_matrix_from_file("/tmp/A.txt", A, M * N)) {
    delete[] A;
    delete[] B;
    return 1;
  }

  if (!read_matrix_from_file("/tmp/B.txt", B, N * K)) {
    delete[] A;
    delete[] B;
    return 1;
  }

  float *C = new float[M * K];
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
  }

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
