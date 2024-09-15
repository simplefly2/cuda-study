#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <cmath>

#define N 1024

#define BLOCK_SIZE 32

#define CHECK(func) do \
{ \
	cudaError_t err = func; \
	if(err != cudaSuccess) \
	{ \
		printf("error: %s, %d \n", cudaGetErrorString(err), __LINE__); \
	}\
} while(0)



template <typename T>
void cpu_histogram(T * input, T * output, int n)
{
	for (int i = 0; i < n; ++i)
	{
		int idx = input[i];

		output[idx]++;
	}
}


__global__ void histogram_v0(int* input, int* output, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n)
	{
		int idx = input[index];

		atomicAdd(&output[idx], 1);
	}
}



int main()
{
	// cpu
	int* input, * output, * gpu_output;

	size_t input_bytes = sizeof(int) * N;
	size_t output_bytes = sizeof(int) * N;

	input = (int*)malloc(input_bytes);
	output = (int*)malloc(output_bytes);
	gpu_output = (int*)malloc(output_bytes);


	for (int i = 0; i < N; ++i)
	{
		input[i] = rand() % N;
		// 显示的对output每个位置进行初始化，否则每个位置是未定义的不确定值，将引起奇怪的结果
		// cpu 直接在这个位置上+1，将得到奇怪的结果
		// gpu 因为atomicAdd 在这个位置 +1， 所有加1的结果最终写到（覆盖）该位置原始值，所有gpu记过不受影响，但显示的初始化，是推荐的行为
		output[i] = 0; 
	}

	
	// gpu:
	int* d_input, * d_output;
	cudaMalloc((void**)&d_input, input_bytes);
	cudaMalloc((void**)&d_output, output_bytes);

	cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_output, output, output_bytes, cudaMemcpyHostToDevice); // 这里增加一个显示的初始化，不进行这一步，gpu的结果也正确


	int grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);

	histogram_v0 <<<grid_size, block_size >>>(d_input, d_output, N);

	cudaMemcpy(gpu_output, d_output, output_bytes, cudaMemcpyDeviceToHost);


	// check:
	cpu_histogram(input, output, N);

	bool error = false;
	for (int i = 0; i < N; ++i)
	{
		if (fabs(gpu_output[i] - output[i]) > 1e-6)

			error = true;

	}


	printf("result: %s \n", error ? "fail" : "pass");


	int offset = 10;
	for (int i = 0; i < 50; ++i)
	{
		printf("cpu: %d, gpu: %d \n", output[offset + i], gpu_output[offset + i]);
	}



	free(input);
	free(output);
	free(gpu_output);
	cudaFree(d_input);
	cudaFree(d_output);


	return 0;
}