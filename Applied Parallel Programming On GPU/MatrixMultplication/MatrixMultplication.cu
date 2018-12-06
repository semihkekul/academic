
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

cudaError_t multiplyWithCuda(int *M_h, int *N_h, int *O_h);

#define BLOCK_SIZE		16
#define MATRIX_M_HEIGHT 2048
#define MATRIX_M_WIDTH	1024
#define MATRIX_N_HEIGHT 1024
#define MATRIX_N_WIDTH  2048

#define MATRIX_O_HEIGHT MATRIX_M_HEIGHT
#define MATRIX_O_WIDTH  MATRIX_N_WIDTH

#define WITH_IO 0
#define WITHOUT_IO 1
#define PRINT_MATRICES 0

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: O = M * N
////////////////////////////////////////////////////////////////////////////////
__global__ void multiplicationKernel(const int *M_d, const int *N_d, int *O_d)
{
	__shared__ int tM_s[BLOCK_SIZE][BLOCK_SIZE];  // tile from M
	__shared__ int tN_s[BLOCK_SIZE][BLOCK_SIZE];  // tile from N
	int tiX = threadIdx.x;// tile index
	int tiY = threadIdx.y;// tile index

	int r = blockIdx.y * BLOCK_SIZE + threadIdx.y;  // row of the element on Output O_d
	int c = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // column of the element on Output O_d

	int sum = 0; // will be calculated value of the corresponding element of O_d

	for (int i = 0; i < gridDim.x; ++i)
	{
		if (r < MATRIX_M_HEIGHT && threadIdx.x + i * BLOCK_SIZE < MATRIX_M_WIDTH)
			tM_s[tiY][tiX] = M_d[r * MATRIX_M_WIDTH + i * BLOCK_SIZE + threadIdx.x];
		else
			tM_s[tiY][tiX] = 0;
		if (c < MATRIX_N_WIDTH && threadIdx.y + i * BLOCK_SIZE < MATRIX_N_HEIGHT)
			tN_s[tiY][tiX] = N_d[c + (i * BLOCK_SIZE + threadIdx.y) * MATRIX_N_WIDTH];
		else
			tN_s[tiY][tiX] = 0;

		__syncthreads();  // wait all threads of the block fill its part of the tile

		for (int j = 0; j < BLOCK_SIZE; ++j)
		{
			sum += tM_s[tiY][j] * tN_s[j][tiX];  // regular inner product
		}
	}
	if (r < MATRIX_M_HEIGHT && c < MATRIX_O_WIDTH)
		O_d[r * MATRIX_N_WIDTH + c] = sum;  // put sum to corresponding place in O
}

int main()
{
	std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
	std::cout << "calling O = M X N .. " << std::endl;
	std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
	int *M_h, *N_h, *O_h;
	M_h = new int[MATRIX_M_HEIGHT * MATRIX_M_WIDTH];
	N_h = new int[MATRIX_N_HEIGHT * MATRIX_N_WIDTH];
	O_h = new int[MATRIX_M_HEIGHT * MATRIX_N_WIDTH];

	for (int i = 0; i < MATRIX_M_HEIGHT * MATRIX_M_WIDTH; ++i)
	{
		M_h[i] = i;
	}
	for (int i = 0; i < MATRIX_N_HEIGHT * MATRIX_N_WIDTH; ++i)
	{
		N_h[i] = i;
	}

	for (int i = 0; i < MATRIX_M_HEIGHT * MATRIX_N_WIDTH; ++i)
	{
		O_h[i] = i;
	}
#if PRINT_MATRICES == 1
	std::cout << "===== M ======" << std::endl;
	for (int it = 0; it < MATRIX_M_HEIGHT * MATRIX_M_WIDTH; ++it)
	{
		if (fmod((double)it, (double)(MATRIX_M_WIDTH)) == 0)
			std::cout << "\\n" << std::endl;
		std::cout << M_h[it] << " ";

	}
	std::cout << "\\n" << std::endl;
	std::cout << "=============" << std::endl;


	std::cout << "===== N ======" << std::endl;
	for (int it = 0; it < MATRIX_N_HEIGHT * MATRIX_N_WIDTH; ++it)
	{
		if (fmod((double)it, (double)(MATRIX_N_WIDTH)) == 0)
			std::cout << "\\n" << std::endl;
		std::cout << N_h[it] << " ";

	}
	std::cout << "\\n" << std::endl;
	std::cout << "=============" << std::endl;
#endif
	// Multiply matrices
	cudaError_t cudaStatus = multiplyWithCuda(M_h, N_h, O_h);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "multiplyWithCuda failed!" << std::endl;
		return 1;
	}
#if PRINT_MATRICES == 1
	std::cout << "===== O ======" << std::endl;
	for (int it = 0; it < MATRIX_M_HEIGHT * MATRIX_N_WIDTH; ++it)
	{
		if (fmod((double)it, (double)(MATRIX_N_WIDTH)) == 0)
			std::cout << "\\n" << std::endl;
		std::cout << O_h[it] << " ";

	}
	std::cout << "\\n" << std::endl;
	std::cout << "=============" << std::endl;
#endif
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
	std::cout << "O = M X N success" << std::endl;
	std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
	system("PAUSE");
	return 0;
}

// Helper function for using CUDA to multiply matrices in parallel.
cudaError_t multiplyWithCuda(int *M_h, int *N_h, int *O_h)
{


	int *M_d, *N_d, *O_d;
	cudaError_t cudaStatus;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid(MATRIX_M_WIDTH / BLOCK_SIZE + (MATRIX_M_WIDTH % BLOCK_SIZE == 0 ? 0 : 1), MATRIX_M_HEIGHT / BLOCK_SIZE + (MATRIX_M_HEIGHT % BLOCK_SIZE == 0 ? 0 : 1), 1);
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&O_d, MATRIX_M_HEIGHT * MATRIX_N_WIDTH * sizeof(int));  // O = M X N
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&M_d, MATRIX_M_WIDTH * MATRIX_M_HEIGHT * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&N_d, MATRIX_N_WIDTH * MATRIX_N_HEIGHT * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaEvent_t start, stop;
	float time;
#if WITH_IO == 1
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(M_d, M_h, MATRIX_M_WIDTH * MATRIX_M_HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(N_d, N_h, MATRIX_N_WIDTH * MATRIX_N_HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

#if WITHOUT_IO == 1
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif


	// Launch a kernel on the GPU with one thread for each element.
	multiplicationKernel << <dimGrid, dimBlock >> >(M_d, N_d, O_d);

	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiplicationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
#if WITHOUT_IO == 1
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\nTime for the kernel: %f ms\n", time);
#endif
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(O_h, O_d, MATRIX_M_HEIGHT * MATRIX_N_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#if WITH_IO == 1
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\nTime for the kernel: %f ms\n", time);
#endif


Error:
	cudaFree(M_d);
	cudaFree(N_d);
	cudaFree(O_d);

	return cudaStatus;
}
