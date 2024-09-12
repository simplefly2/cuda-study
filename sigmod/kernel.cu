#include "stdio.h"
#include "stdlib.h"
#include "cuda_runtime.h"
#include <cmath>



#define CHECK(func) do \
{  \
	cudaError_t err = func; \
	if(err != cudaSuccess) \
	{ \
		printf("error, %s, %d \n", cudaGetErrorString(err), __LINE__); \
	} \
} while (0);


#define M 1024
#define N 768

#define BLOCK_SIZE 256

#define FLOAT4(value) *(float4*)(&value)


/**
* sigmod: 
* 
* y(xi) = 1 / (1+ exp(-xi)) 
* 
*/


/**
* v0: base
* 
* v1: float4
* (假设数据能被4整除)
*/

template <typename T>
void cpu_sigmod(T * input, T* output, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			T x = (T)input[i * n + j];
			output[i * n + j] = 1.0 / (1 + (expf(-x)));
		}
	}
}



// v0: 一个线程计算一个数：
template <typename T>
__global__ void sigmoid_v0(T* input, T* output, int m, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m*n)
	{
		T x = input[index];
		output[index] = 1.0f / (1.0f + expf(-x));
	}
}

// v1: float4
__global__ void sigmoid_v1(float* input, float* output, int m, int n)
{
	int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

	if (index < m * n)
	{
		float4 val = FLOAT4(input[index]);
		float4 res;

		res.x = 1.0f / (1.0f + expf(-val.x));
		res.y = 1.0f / (1.0f + expf(-val.y));
		res.z = 1.0f / (1.0f + expf(-val.z));
		res.w = 1.0f / (1.0f + expf(-val.w));

		// *(float4*)(&output[index]) = res;
		FLOAT4(output[index]) = res;
	}
}



int main()
{
	float* input, * output, * gpu_output;

	size_t input_bytes = sizeof(float) * M * N;
	size_t output_bytes = sizeof(float) * M * N;
	
	input = (float*)malloc(input_bytes);
	output = (float*)malloc(output_bytes);
	gpu_output = (float*)malloc(output_bytes);


	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			input[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
		}
	}

	cpu_sigmod(input, output, M, N);



	// gpu
	float* d_input, * d_output;
	cudaMalloc((void**)&d_input, input_bytes);
	cudaMalloc((void**)&d_output, output_bytes);

	cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

	int grid_x = (M*N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);

	/*sigmoid_v0<<<grid_size, block_size>>>(d_input, d_output, M, N);*/
	sigmoid_v1<<<grid_size, block_size >>>(d_input, d_output, M, N);

	cudaMemcpy(gpu_output, d_output, output_bytes, cudaMemcpyDeviceToHost);


	// check:
	bool error = false;
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if (fabs(gpu_output[i * N + j] - output[i * N + j]) > 1e-6)

				error = true;
		}
	}



	printf("result: %s \n", error ? "fail" : "pass");


	int offset = 100;
	for (int i = 0; i < 50; ++i)
	{
		printf("cpu: %.6f, gpu: %.6f \n", output[offset + i], gpu_output[offset + i]);
	}


	free(input);
	free(output);
	free(gpu_output);
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}
