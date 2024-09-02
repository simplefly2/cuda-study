#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

# define CHECK(func) do \
{ \
	cudaError_t err = func; \
	if (err != cudaSuccess) \
	{                 \
		printf("error: %s, %d \n", cudaGetErrorString(err), __LINE__); \
	} \
} while(0)



#define M 1024
#define N 768

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

// input, shape(M, N)
// output, shape(M, N)
// mean = sum(ni) / N
// std = sqrt(sum((mean-xi)*(mean-xi)) / N)

template <typename T>
void cpu_layernorm(T * input, T * output, T g, T b, int m, int n)
{
	for (unsigned i = 0; i < m; ++i)
	{
		T mean = 0.0;
		T std = 0.0;

		// mean
		T mean_sum = (T)0.0;
		for (unsigned j = 0; j < n; ++j)
		{
			mean_sum += input[i * n + j];
		}

		mean = mean_sum / n;

		// std
		T std_sum = (T)0.0;
		for (unsigned j = 0; j < n; ++j)
		{
			std_sum += ((mean - input[i * n + j]) * (mean - input[i * n + j]));
		}

		std = sqrtf(std_sum / n);

		// output: (xi-mean) / std;
		for (unsigned k = 0; k < n; ++k)
		{
			output[i * n + k] = g*(input[i * n + k] - mean) / (std + 1e-6) + b;
		}
	}
}


// 每行的归一化是独立的，所以可以并行
// 一个线程处理一行数据
template <typename T>
__global__ void layerNorm_kernel(T* input, T* output, T g, T b, int m, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int tid = threadIdx.x;

	T* data;
	if (index < m)
	{
		data = input + index * n;
	}

	// mean:
	T mean_sum = (T)0.0;
	for (unsigned j = 0; j < n; ++j)
	{
		mean_sum += data[j];
	}

	T mean = mean_sum / n;

	T std_sum = (T)0.0;
	for (unsigned j = 0; j < n; ++j)
	{
		std_sum += (data[j] - mean) * (data[j] - mean);
	}

	T std = sqrtf(std_sum / n);

	for (unsigned j = 0; j < n; ++j)
	{
		output[index * n + j] = g*(data[j]-mean)/(std+1e-6) + b;
	}
}


// warp reduce:
template <typename T>
__device__ T warpReduce(T val)
{
	for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 2)
	{
		val += __shfl_down_sync(0xffffffff, val, offset);
	}

	return __shfl_sync(0xffffffff, val, 0);
}



// 每个 warp 处理一行数据：
template <typename T>
__global__ void layerNorm_warp(T* input, T* output, T g, T b, int m, int n)
{
	// mean:
	int warpid = threadIdx.x / WARP_SIZE;
	int laneid = threadIdx.x % WARP_SIZE;
	int bid = blockIdx.x;
	int warp_num = blockDim.x / WARP_SIZE;

	int row = bid * warp_num + warpid;
	if(row < m)
	{
		T* inp = input + row * n;
		T mean = (T)0.0;
		for (int i = laneid; i < n; i += WARP_SIZE)
		{
			mean += inp[i];
		}

		mean = warpReduce<T>(mean);

		mean = mean / n;

		// __syncthreads();
		
		// std:
		T std = (T)0.0;
		for (int i = laneid; i < n; i += WARP_SIZE)
		{
			std += (inp[i] - mean) * (inp[i] - mean);
		}

		std = warpReduce<T>(std);

		std = sqrtf(std / n);

		// __syncthreads();
		
		// layernorm:
		
		for (int i = laneid; i < n; i += WARP_SIZE)
		{
			output[row*n+i] = g*(inp[i] - mean) / (std + 1e-6) + b;
		}

	}

}
//
//// block reduce
//__global__ void layerNorm_kernel_1(float* input, float* output, float g, float b, int m, int n)
//{
//
//
//}
//
//// shared 加速 reduce操作：
//// reduce + vec4
//__global__ void layerNorm_kernel_3(float* input, float* output, float g, float b, int m, int n)
//{
//
//
//}




int main()
{
	// cpu:
	float* input, * output;
	float g, b;

	size_t data_bytes = sizeof(float) * (M * N);

	input = (float*)malloc(data_bytes);
	output = (float*)malloc(data_bytes);

	for (unsigned i = 0; i < M; ++i)
	{
		for (unsigned j = 0; j < N; ++j)
		{
			input[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
		}
	}

	g = 0.1;
	b = 0.1;
	cpu_layernorm(input, output, g, b, M, N);

	//for (unsigned i = 0; i < 20; ++i)
	//{
	//	printf("\n");
	//	for (unsigned j = 0; j < 10; ++j)
	//	{
	//		printf("%.6f %s", output[i*N+j], "  ");
	//	}
	//}

	// gpu
	float* gpu_output;
	gpu_output = (float*)malloc(data_bytes);

	float* d_input, * d_output;
	CHECK(cudaMalloc((void**)&d_input, data_bytes));
	CHECK(cudaMalloc((void**)&d_output, data_bytes));

	CHECK(cudaMemcpy(d_input, input, data_bytes, cudaMemcpyHostToDevice));

	//size_t grid_x = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//dim3 grid_size(grid_x);
	//dim3 block_size(BLOCK_SIZE);
	//layerNorm_kernel<<<grid_size, block_size >>>(d_input, d_output, g, b, M, N);

	size_t grid_x = (M + (BLOCK_SIZE/WARP_SIZE) - 1) / (BLOCK_SIZE/WARP_SIZE);
	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);
	layerNorm_warp<<<grid_size, block_size>>>(d_input, d_output, g, b, M, N);

	cudaMemcpy(gpu_output, d_output, data_bytes, cudaMemcpyDeviceToHost);

	// check:
	bool error = false;
	for (unsigned i = 0; i < M; ++i)
	{
		for (unsigned j = 0; j < N; ++j)
		{
			if (fabs(gpu_output[i * N + j] - output[i * N + j]) > 1e-6)
			{
				error = true;
			}
		}
	}

	printf("result: %s \n", error ? "fail" : "pass");

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			printf("gpu: %.8f, cpu: %.8f \n", gpu_output[i * N + j], output[i * N + j]);
		}
	}



	free(input);
	free(output);
	free(gpu_output);
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}