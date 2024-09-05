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


#define BLOCK_SIZE 256

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


template <typename T>
__global__ void marixMulKernrl0(T* A, T* B, T* C, int m, int n, int k)
{




}


int main()
{
	// cpu:
	float* A, *B, *C;

	size_t A_bytes = sizeof(float) * M * N;
	size_t B_bytes = sizeof(float) * N * K;
	size_t C_bytes = sizeof(float) * M * K;

	A = (float*)malloc(A_bytes);
	B = (float*)malloc(B_bytes);
	C = (float*)malloc(C_bytes);

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
	float* gpu_C;
	float* d_A, * d_B, * d_C;

	cudaMalloc((void**)&d_A, A_bytes);
	cudaMalloc((void**)&d_B, B_bytes);
	cudaMalloc((void**)&d_C, C_bytes);


	cudaMemcpy(d_A, A, A_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, B_bytes, cudaMemcpyHostToDevice);

	
	int grid_x = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 grid_size(grid_x, grid_y);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	marixMulKernrl0<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

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
		printf("cpu value: %.6f, gpu value: %.6f \n", C[offset + i], d_C[offset+i]);
	}


	free(A);
	free(B);
	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;

}