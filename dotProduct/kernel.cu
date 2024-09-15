#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "cuda_runtime.h"


#define N 2048

#define BLOCK_SIZE 256




template <typename T>
void cpu_dotProduct(T* vec1, T* vec2, T* sum, int n)
{
	*sum = (T)0.0f;
	for (int i = 0; i < n; ++i)
	{
		*sum += (vec1[i] * vec2[i]);
	}
}



/**
*
*	v0:
*
*/
template <typename T>
__global__ void dotProductV0(T* vec1, T* vec2, T* sum, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n)
	{
		T temp_sum = (T)0.0f;
		temp_sum = vec1[index] * vec2[index];

		atomicAdd(sum, temp_sum);
	}

}



/**
*  
*	shared + block reduce
* 
*/
template <typename T>
__global__ void dotProductV1(T* vec1, T* vec2, T* sum, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	extern __shared__ T sdata[];

	sdata[tid] = index < n ? vec1[index] * vec2[index] : (T)0.0f;

	// block reduce:
	for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
	{
		if (tid < s)

			sdata[tid] = sdata[tid] + sdata[tid + s];

		__syncthreads();
	}


	// 累加各个block sum值：
	if (tid == 0)

		atomicAdd(sum, sdata[tid]);

}




/**
*
*	V2: float4 + shared + block reduce
*
*/

//__global__ void dotProductV2(float* vec1, float* vec2, float* sum, int n)
//{
//	
//
//}






int main()
{
	float* vec1, * vec2, * sum, * gpu_sum;

	size_t vec_bytes = sizeof(float) * N;
	size_t sum_bytes = sizeof(float);

	vec1 = (float*)malloc(vec_bytes);
	vec2 = (float*)malloc(vec_bytes);
	sum = (float*)malloc(sum_bytes);
	gpu_sum = (float*)malloc(sum_bytes);


	for (int i = 0; i < N; ++i)
	{
		vec1[i] = static_cast<float>(rand()) / RAND_MAX;
		vec2[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	
	cpu_dotProduct(vec1, vec2, sum, N);

	// gpu:
	float* d_vec1, * d_vec2, * d_sum;

	cudaMalloc((void**)&d_vec1, vec_bytes);
	cudaMalloc((void**)&d_vec2, vec_bytes);
	cudaMalloc((void**)&d_sum, sum_bytes);


	cudaMemcpy(d_vec1, vec1, vec_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec2, vec2, vec_bytes, cudaMemcpyHostToDevice);


	//size_t grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//dim3 grid_size(grid_x);
	//dim3 block_size(BLOCK_SIZE);

	//dotProductV0 <<<grid_size, block_size>>>(d_vec1, d_vec2, d_sum, N);
	
	size_t grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);

	dotProductV1<<< grid_size, block_size, sizeof(float)*BLOCK_SIZE >>>(d_vec1, d_vec2, d_sum, N);

	cudaMemcpy(gpu_sum, d_sum, sum_bytes, cudaMemcpyDeviceToHost);


	// check:
	bool error = false;
	if (fabs(*sum - *gpu_sum) > 1e-3)

		error = true;

	printf("result: %s \n", error ? "fail" : "pass");

	printf("cpu: %.6f , gpu: %.6f \n", *sum, *gpu_sum);


	free(vec1);
	free(vec2);
	free(sum);
	cudaFree(d_vec1);
	cudaFree(d_vec2);
	cudaFree(d_sum);

	return 0;
}