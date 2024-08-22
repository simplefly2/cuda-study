

// element wise:
// vector add
// relu
// sigomd
// ...

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define CHECK(func) {                                                     \
	cudaError_t err = func;                                               \
	if (err != cudaSuccess)                                               \
	{                                                                     \
		printf("error: %s, line: %d", cudaGetErrorString(err), __LINE__);  \
 	}                                                                     \
}                                                                         \

#define BLOCK_SIZE 32

#define N 1000000

#define FLOAT4(value) *((float4*)(&value))


__global__ void relu_kernel(float* input, float* output, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < n)
		output[index] = input[index] > 0 ? input[index] : 0;
}

__global__ void relu_kernel_vec4(float* input, float* output, int n)
{
	int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

	if (index < n)
	{
		float4 tmpx = FLOAT4(input[index]);
		float4 tmpy;
		tmpy.x = fmaxf(0, tmpx.x);
		tmpy.y = fmaxf(0, tmpx.y);
		tmpy.z = fmaxf(0, tmpx.z);
		tmpy.w = fmaxf(0, tmpx.w);

		FLOAT4(output[index]) = tmpy;
	}
}


void relu_cpu(float* input, float* output, int n)
{
	for (int i = 0; i < n; ++i)
	{
		output[i] = input[i] > 0 ? input[i] : 0;
	}
}

int main()
{
	float* input, *cpu_output;
	float* gpu_output;

	size_t input_bytes = sizeof(float) * N;
	size_t output_bytes = sizeof(float) * N;

	input = (float*)malloc(input_bytes);
	cpu_output = (float*)malloc(output_bytes);
	gpu_output = (float*)malloc(output_bytes);

	for (int i = 0; i < N; ++i)
	{
		input[i] = (float)(rand() % 1024);
	}


	float* d_input, * d_output;
	

	CHECK(cudaMalloc((void**)&d_input, input_bytes));
	CHECK(cudaMalloc((void**)&d_output, output_bytes));

	CHECK(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));

	//unsigned int grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_x = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);

	//relu_kernel<<<grid_size, block_size >>>(d_input, d_output, N);
	relu_kernel_vec4 <<<grid_size, block_size >>> (d_input, d_output, N);
	relu_cpu(input, cpu_output, N);

	CHECK(cudaMemcpy(gpu_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

	// check:

	for (int i = 0; i < N; ++i)
	{
		if (fabs(gpu_output[i] - cpu_output[i]) > 1e-10)
		{
			printf("error, index: %d, gpu: %.2f, cpu: %.2f", i, gpu_output[i], cpu_output[i]);
			exit(1);
		}
	}
	printf("pass");

	free(input);
	free(cpu_output);
	cudaFree(d_input);
	cudaFree(d_output);

}