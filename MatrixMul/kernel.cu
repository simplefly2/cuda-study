#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "cuda_runtime.h"


#define CHECK(func) do \
{ \
	cudaError_t err = func; \
	if(err != cudaSuccess) \
	{ \
		printf("error: %s, %d \n", cudaGerErrorString(err), __LINE__); \
	} \
} while(0);


#define BLOCK_SIZE 32

#define M 1024
#define N 768
#define K 2048


/*
* A: M x N;
* B: N x K;
* 
* A * B --> C (M, K)
* 
*/


template <typename T>
void cpu_matrixMul(T * A, T * B, T* C, int m, int n, int k)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			T sum = 0.0f;
			for (int t = 0; t < n; ++t)

				sum += A[i * n + t] * B[t * k + j];
			
			C[i * k + j] = sum;
		}
	}
}

// kerneo 0: 一个线程处理 C 矩阵一个对应位置的数据
template <typename T>
__global__ void matrixMulKernrl0(T* A, T* B, T* C, int m, int n, int k)
{
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int row = bidy * blockDim.y + tidy;
	int col = bidx * blockDim.x + tidx;

	if (row < m && col < k)
	{
		T sum = (T)0.0f;
		for (int s = 0; s < n; ++s)
		{
			sum += A[row * n + s] * B[s * k + col];
		}

		C[row * k + col] = sum;
	}

}

// kernel1: 
// 结果矩阵分块计算，一个block负责计算一个方块的数据
// block: (BLOCK_SIZE, BLOCK_SIZE)   分块大小
//template <typename T>
//__global__ void matrixMulKernel1(T* A, T* B, T* C, int m, int n, int k)
//{
//	// 分配shared内存大小为一个block加载的数据大小：
//	__shared__ T s_a[BLOCK_SIZE][BLOCK_SIZE];
//	__shared__ T s_b[BLOCK_SIZE][BLOCK_SIZE];
//
//	int bidx = blockIdx.x;
//	int bidy = blockIdx.y;
//	int tidx = threadIdx.x;
//	int tidy = threadIdx.y;
//
//	int row = bidy * blockDim.y + tidy;
//	int col = bidx * blockDim.x + tidx;
//
//	// 窗口滑动，分块计算
//	for (int i = 0; i < n / BLOCK_SIZE; ++i)
//	{
//		if (row < m && col < k)
//		{
//			// load data:
//			s_a[tidy][tidx] = A[row*n + tidx + i*BLOCK_SIZE];
//			s_b[tidy][tidx] = B[(tidy + i * BLOCK_SIZE) * k + col];
//
//			__syncthreads();
//
//			// 子方块计算：
//			T sum = (T)0.0f;
//			for (int s = 0; s < BLOCK_SIZE; ++s)
//				sum += s_a[tidy][s] * s_b[s][tidx];
//
//			C[row * k + col] = sum;
//		}
//	}
//}


template <typename T>
__global__ void matrixMulKernel1(T* A, T* B, T* C, int m, int n, int k)
{
	__shared__ T s_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ T s_b[BLOCK_SIZE][BLOCK_SIZE];

	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int row = bidy * BLOCK_SIZE + tidy;
	int col = bidx * BLOCK_SIZE + tidx;

	T sum = (T)0.0f;

	// 确保线程不会越界  
	if (row < m && col < k)
	{
		// 窗口滑动，分块计算  
		for (int i = 0; i < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
		{
			int a_idx = row * n + tidx + i * BLOCK_SIZE;
			int b_idx = (tidy + i * BLOCK_SIZE) * k + col;

			// 加载数据到shared memory  
			s_a[tidy][tidx] = (a_idx < m * n) ? A[a_idx] : (T)0.0f;
			s_b[tidy][tidx] = (b_idx < n * k) ? B[b_idx] : (T)0.0f;

			__syncthreads();

			// 子方块计算  
			for (int s = 0; s < BLOCK_SIZE; ++s)
				sum += s_a[tidy][s] * s_b[s][tidx];

			__syncthreads(); // 理论上这里不需要，因为每个线程都在独立计算sum  
		}

		// 将结果写回全局内存  
		C[row * k + col] = sum;
	}
}


int main()
{
	// cpu:
	float* A, * B, * C, *gpu_C;

	size_t A_bytes = sizeof(float) * M * N;
	size_t B_bytes = sizeof(float) * N * K;
	size_t C_bytes = sizeof(float) * M * K;

	A = (float*)malloc(A_bytes);
	B = (float*)malloc(B_bytes);
	C = (float*)malloc(C_bytes);
	gpu_C = (float*)malloc(C_bytes);

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)

			A[i * N + j] = static_cast<float>(rand() % 10);
	}

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < K; ++j)

			B[i * K + j] = static_cast<float>(rand() % 10);
	}

	cpu_matrixMul(A, B, C, M, N, K);


	// gpu;
	float* d_A, * d_B, * d_C;

	cudaMalloc((void**)&d_A, A_bytes);
	cudaMalloc((void**)&d_B, B_bytes);
	cudaMalloc((void**)&d_C, C_bytes);


	cudaMemcpy(d_A, A, A_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, B_bytes, cudaMemcpyHostToDevice);

	
	//int grid_x = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//dim3 grid_size(grid_x, grid_y);
	//dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	//matrixMulKernrl0<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

	int grid_x = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 grid_size(grid_x, grid_y);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	matrixMulKernel1<<<grid_size, block_size >>>(d_A, d_B, d_C, M, N, K);


	cudaMemcpy(gpu_C, d_C, C_bytes, cudaMemcpyDeviceToHost);


	// check:
	bool error = false;
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < K; ++j)
		{
			if (fabs(gpu_C[i * K + j] - C[i * K + j]) > 1e-6)

				error = true;
		}
	}


	printf("result: %s \n", error ? "fail" : "pass");


	int offset = 100;
	for (int i = 0; i < 20; ++i)
	{
		printf("cpu value: %.6f, gpu value: %.6f \n", C[offset + i], gpu_C[offset+i]);
	}


	free(A);
	free(B);
	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;

}