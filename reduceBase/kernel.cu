#include "stdio.h"
#include "stdlib.h"
#include <cmath>

#include "cuda_runtime.h"


#define M 1024
#define N 768

#define WARP_SIZE 32
#define BLOCK_SIZE 256


/*
	
	计算矩阵 M* N 每行之和

*/ 

template <typename T>
void cpu_MatrixRowSum(T* input, T* output, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		T sum = 0;
		for (int j = 0; j < n; ++j)
			sum += input[i * n + j];
		
		output[i] = sum;
	}
}



template <typename T>
__device__ T warpReduce(T val)
{
	for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xffffffff, val, offset);

	return val;
}


// 每个warp归约一行数据：
template <typename T>
__global__ void matrixRowSum(T* input, T* output, int m, int n)
{
	int warpid = threadIdx.x / WARP_SIZE;
	int laneid = threadIdx.x % WARP_SIZE;
	int warp_num = blockDim.x / WARP_SIZE;

	int row = blockIdx.x * warp_num + warpid;
	if (row < m)
	{
		T* inp = input + blockIdx.x * warp_num * n;

		T sum = (T)0.0f;
		for (int i = laneid; i < n; i += WARP_SIZE)
		{
			sum += inp[i];
		}

		sum = warpReduce<T>(sum);


		if (laneid == 0)
			output[row] = sum;
	}

}


int main()
{
	// cpu:
	int* input, * output, *gpu_output;
	size_t input_bytes = sizeof(int) * M * N;
	size_t output_bytes = sizeof(int) * M;

	input = (int*)malloc(input_bytes);
	output = (int*)malloc(output_bytes);
	gpu_output = (int*)malloc(output_bytes);

	for (unsigned int i = 0; i < M; ++i)
	{
		for (unsigned int j = 0; j < N; ++j)
		{
			input[i * N + j] = rand() % 1024;
		}
	}

	cpu_MatrixRowSum(input, output, M, N);

	// gpu:
	int* d_input, * d_output;
	cudaMalloc((void**)&d_input, input_bytes);
	cudaMalloc((void**)&d_output, output_bytes);

	cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

	int grid_x = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	printf("1111111111grid_x: %d \n", grid_x);
	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);

	matrixRowSum<<<grid_size, block_size>>>(d_input, d_output, M, N);

	cudaMemcpy(gpu_output, d_output, output_bytes, cudaMemcpyDeviceToHost);


	// check:
	bool error = false;
	for (int i = 0; i < M; ++i)
	{
		if (fabs(output[i] - gpu_output[i]) > 1e-6)

			error = true;
	}

	printf("result: %s \n", error ? "fail" : "pass");

	int offset = 20;
	for (int i = offset; i < offset+20; ++i)
		printf("gpu: %d, cpu: %d \n", gpu_output[i], output[i]);


	// free
	free(input);
	free(output);
	free(gpu_output);
	cudaFree(d_input);
	cudaFree(d_output);


	return 0;

}