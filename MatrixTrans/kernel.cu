#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <math.h>

#define BLOCK_SIZE 16

void matrix_transpose_cpu(int* A, int* res, int m, int n)
{
	for (int y = 0; y < n; ++y)
	{
		for (int x = 0; x < m; ++x) {
			
			res[y * m + x] = A[x * n + y];
		}
	}
}

__global__ void matrix_transpose_cuda(int* A, int* res, int m, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < n && x < m)
	{
		res[y * m + x] = A[x * n + y];
	}
	
}

int main()
{
	const unsigned int M = 10000;
	const unsigned int N = 20000;

	unsigned int nBytes = M * N * sizeof(int);

	// allocate memory on cpu and init data
	int* h_A, * h_res_cpu, * h_res_gpu;

	h_A = (int*)malloc(nBytes);
	h_res_cpu = (int*)malloc(nBytes);

	for (unsigned int y = 0; y < M; ++y)
	{
		for (unsigned int x = 0; x < N; ++x)
		{
			h_A[y*N + x] = rand() % 1024;
		}
		
	}

	matrix_transpose_cpu(h_A, h_res_cpu, M, N);

	// allocate memory on gpu and copy data from cpu to gpu
	int* d_A, * d_res;
	cudaMalloc((void**)&d_A, nBytes);
	cudaMalloc((void**)&d_res, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	
	// launch cuda kernel to calc
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrix_transpose_cuda<<<gridSize, blockSize>>>(d_A, d_res, m, n);

	// copy gpu res to cpu:
	cudaMemcpy(h_res_gpu, d_res, nBytes, cudaMemcpyDeviceToHost);

	// check result between gpu and cpu:
	bool error = false;
	for (int y = 0; y < N; ++y)
	{
		for (int x = 0; x < M; ++x)
		{
			if (fabs(h_res_cpu[y * M + x] - h_res_gpu[y * M + x] > 1e-10))
			{
				error = true;
				break;
			}
		}
	}

	printf("cuda result is: %s \n", error ? "Failed" : "Pass");

	return 0;
}