#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "error.cuh"

#include <math.h>

/*
笔记要点：
	1. cuda 编程：
	- 异步执行，一个kernel 对应一个grid,grid内的全部线程都去执行这个kernel
	- 程序的修饰符
	- 基本步骤，几个步骤
	- 程序的基本结构，grid, block ，全局index 
	- kernel的基本写法和调用（越界判断， grid, block 的划分）
	
	2. cuda API返回类型：cudaError_t，可以结合
		cudaError_t err = cudaGetLastError()
		char* err_info = cudaGetErrorString(err)
		检查 api 的执行情况
	
	3. cuda event 的使用：
	- cudaEvent_t event;
	- cudaEventCreate(&event);
	- cudaEventRecord(event);
	- cudaEventDestroy(event);

	4. 同步操作：
	- cudaMemcpy 内在包含了同步，
	- 显示使用 cudaDeviceSynchronize() 在host 等待 gpu 的执行完成
		
*/


#define BLOCK_SIZE 256

__global__ void vecAdd_cuda(float* d_va, float* d_vb, float* d_res, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < N)
		d_res[index] = d_va[index] + d_vb[index];
}


void vecAdd_cpu(float* h_va, float* h_vb, float* h_res, int N)
{
	for (int i = 0; i < N; ++i)
	{
		h_res[i] = h_va[i] + h_vb[i];
	}
}


int main()
{
	const unsigned int N = 100000000;
	size_t nbytes = N * sizeof(float);

	// allocate memory on cpu and init data:
	float* h_va, * h_vb, * h_res_cpu, * h_res_gpu;
	h_va = (float*)malloc(nbytes);
	h_vb = (float*)malloc(nbytes);
	h_res_cpu = (float*)malloc(nbytes);
	h_res_gpu = (float*)malloc(nbytes);

	for (int i = 0; i < N; ++i)
	{
		h_va[i] = rand() % 1024;
		h_vb[i] = rand() % 1024;
	}

	// allocate memory on gpu and cpy data to gpu:
	cudaEvent_t start, stop_gpu, stop_cpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_gpu);
	cudaEventCreate(&stop_cpu);

	cudaEventRecord(start, 0);

	float* d_va, * d_vb, * d_res;
	cudaMalloc((void**)&d_va, nbytes);
	cudaMalloc((void**)&d_vb, nbytes);
	cudaMalloc((void**)&d_res, nbytes);

	cudaMemcpy(d_va, h_va, nbytes, cudaMemcpyHostToDevice);
	cudaError_t err = cudaMemcpy(d_vb, h_vb, nbytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed %s", cudaGetErrorString(err));
	}

	// launch a cuda kernel:
	unsigned int grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 gridSize(grid_x);
	dim3 blockSize(BLOCK_SIZE);

	vecAdd_cuda<<<gridSize, blockSize>>>(d_va, d_vb, d_res, N);

	cudaDeviceSynchronize();

	cudaError_t error_code = cudaGetLastError();
	if (error_code != cudaSuccess)
	{
		const char* error_info = cudaGetErrorString(error_code);
		fprintf(stderr, "cuda get error: %s", error_info);
		exit(EXIT_FAILURE);
	}

	// cpy data from gpu to cpu:
	cudaMemcpy(h_res_gpu, d_res, nbytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop_gpu, 0);
	cudaEventSynchronize(stop_gpu);


	// calc on cpu and check result:
	vecAdd_cpu(h_va, h_vb, h_res_cpu, N);

	cudaEventRecord(stop_cpu, 0);
	cudaEventSynchronize(stop_cpu);

	float elapsed_time_gpu, elapsed_time_cpu;
	cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu);
	cudaEventElapsedTime(&elapsed_time_cpu, stop_gpu, stop_cpu);

	bool error = false;
	for (int i = 0; i < N; ++i)
	{
		if (fabs(h_res_gpu[i] - h_res_cpu[i]) > 1e-10)
			error = true;
	}


	printf("the result on gpu is: %s \n", error ? "failed" : "pass");
	printf("cpu elapsed time:%.2f ms\n", elapsed_time_cpu);
	printf("gpu elapsed time:%.2f ms\n", elapsed_time_gpu);

	cudaEventDestroy(start);
	cudaEventDestroy(stop_gpu);
	cudaEventDestroy(stop_cpu);

	cudaFree(d_va);
	cudaFree(d_vb);
	cudaFree(d_res);

	free(h_va);
	free(h_vb);
	free(h_res_cpu);
	free(h_res_gpu);

	return 0;

}