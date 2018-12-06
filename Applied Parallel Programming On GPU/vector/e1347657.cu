#include <iostream>
#include <stdio.h>
#include <cuda.h>
//cuda definitions
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

#define  VECTOR_SIZE_INNER_PRODUCT  512


	//device function
	__global__ void innerProduct(int* x, int* y, int *result)
	{
		int idx = threadIdx.x; // 1 huge 1D block
		__shared__ int sharedMemory[VECTOR_SIZE_INNER_PRODUCT];
		 sharedMemory[idx] = x[idx] * y[idx]; // make products first
		__syncthreads();
		if (threadIdx.x == 0) 
		{
			int total = 0;
			for (int i = 0; i < VECTOR_SIZE_INNER_PRODUCT; i++)
			{
				total += sharedMemory[i];
			}
			*result = total;
		}

	}


/****************************
Host function to prepare and call device function
****************************/
void callInnerProduct()
{
	std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
	std::cout << "Calling inner product.. " << std::endl;
	std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
	int *Vec1_h, *Vec2_h, *Vec1_d, *Vec2_d;
	int *result_d, *result_h;
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int size = VECTOR_SIZE_INNER_PRODUCT * sizeof(int);
	
	Vec1_h = (int*)malloc(size);
	Vec2_h = (int*)malloc(size);
	result_h = (int*)malloc(sizeof(int));

	for (int i = 0; i < VECTOR_SIZE_INNER_PRODUCT; ++i)
	{
		Vec1_h[i] = i;
		Vec2_h[i] = i;
	}

	cudaMalloc((void**)&Vec1_d, size);
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	cudaMalloc((void**)&Vec2_d, size);
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	cudaMalloc((void**)&result_d, sizeof(int));
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	


	

	cudaMemcpy(Vec1_d, Vec1_h, size, cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	cudaMemcpy(Vec2_d, Vec2_h, size, cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;

	cudaEventRecord(start, 0);
	////////////////////////////////////////////////////////////////////////////////////////////
	innerProduct << < 1, VECTOR_SIZE_INNER_PRODUCT >> >(Vec1_d, Vec2_d, result_d);
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);


	cudaMemcpy(result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost); // blocks host
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	cudaMemcpy(Vec1_h, Vec1_d, size, cudaMemcpyDeviceToHost); // blocks host
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	cudaMemcpy(Vec2_h, Vec2_d, size, cudaMemcpyDeviceToHost); // blocks host
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;

	

	printf("Time for the kernel: %f ms\n", time);
	std::cout << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl;
	printf("result is %d\n", *result_h);

	std::cout << "==== Vector ====" << std::endl;
	for (int it = 0; it < VECTOR_SIZE_INNER_PRODUCT; ++it)
	{
		std::cout << Vec1_h[it] << " ";
	}
	std::cout << "\\n" << std::endl;
	std::cout << "=============" << std::endl;
	std::cout << "==== Vector ====" << std::endl;
	for (int it = 0; it < VECTOR_SIZE_INNER_PRODUCT; ++it)
	{
		std::cout << Vec2_h[it] << " ";
	}
	std::cout << "\\n" << std::endl;
	std::cout << "=============" << std::endl;
	free(Vec1_h);
	free(Vec2_h);
	free(result_h);
	cudaFree(Vec1_d);
	cudaFree(Vec2_d);
	cudaFree(result_d);
}


namespace matrix
{
	/****************************************
	MATRIX struct row major order
	*****************************************/
	typedef struct
	{
		int m; //row count
		int n; //column count
		float* data; //data
	} MATRIX;

	/****************************************
	@return N/A
	****************************************/
	__global__ void matrix_summation(MATRIX X, MATRIX Y, int M, int N)
	{
		// 2D column and 2D row
		int idxRow = blockIdx.y * blockDim.y + threadIdx.y;
		int idxColumn = blockIdx.x * blockDim.x + threadIdx.x;
		int idx = idxRow * N + idxColumn;
		if (idxRow < M && idxColumn < N)
		{
			X.data[idx] = Y.data[idx] + X.data[idx];
		}
	}



		/****************************************
		fills the content of the matrix
		@param matrix
		@return success
		****************************************/
		bool fillMatrix(MATRIX& matrix)
		{
			if (matrix.data == NULL || matrix.m <= 0 || matrix.n <= 0)
			{
				std::cout << "Invalid Matrix! " << __LINE__ << std::endl;
				return false;
			}
			for (int i = 0; i < matrix.m * matrix.n; ++i)
			{
				matrix.data[i] = (float)i;
			}
			return true;
		}
		/****************************************
		prints the content of the matrix
		@param matrix
		@return success
		****************************************/
		bool dumpMatrix(MATRIX& matrix)
		{
			if (matrix.data == NULL || matrix.m <= 0 || matrix.n <= 0)
			{
				std::cout << "Invalid Matrix! " << __LINE__ << std::endl;
				return false;
			}
			std::cout << "==== Matrix ====" << std::endl;
			
			for (int i = 0; i < matrix.m * matrix.n; ++i)
			{
				if (fmod((double)i, (double)(matrix.n)) == 0)
					std::cout << "\\n"<<std::endl;
				std::cout << std::fixed<<matrix.data[i] << " ";

			}
			std::cout << "\\n" << std::endl;
			std::cout << "=============" << std::endl;
			return true;
		}
		/****************************************
		@param M
		@param N
		@param matrix
		@return success
		*****************************************/
		bool allocateMatrixForDevice(const int M, const int N, MATRIX& matrix)
		{
			if (M <= 0 || N <= 0)
			{
				std::cout << "Invalid M or N! " << __LINE__ << std::endl;
				return false;
			}
			matrix.m = M;
			matrix.n = N;
			if (cudaMalloc((void **)&matrix.data, sizeof(float) * M *N) != cudaError::cudaSuccess)
			{
				std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ << std::endl;
				return false;
			}
			return true;
		}
		/****************************************
		@param M
		@param N
		@param matrix
		@return success
		*****************************************/
		bool allocateMatrixForHost(const int M, const int N, MATRIX& matrix)
		{
			if (M <= 0 || N <= 0)
			{
				std::cout << "Invalid M or N! " << __LINE__ << std::endl;
				return false;
			}
			matrix.m = M;
			matrix.n = N;
			matrix.data = new float[M*N];
			if (matrix.data == NULL)
			{
				std::cout << "Memory allocation error! " << __LINE__ << std::endl;
				return false;
			}
			return true;
		}
		/****************************************
		*****************************************/
		bool copyHostToDevice(MATRIX& D, MATRIX& H)
		{
			if (H.data == NULL || H.m <= 0 || H.n <= 0 || D.m <= 0 || D.n <= 0)
			{
				std::cout << "Invalid Matrix! " << __LINE__ << std::endl;
				return false;
			}
			if (cudaMemcpy(D.data, H.data, H.m * H.n * sizeof(float), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
			{
				std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ << std::endl;
				return false;
			}
			return true;
		}
		/****************************************
		*****************************************/
		bool copyDeviceToHost(MATRIX& H, MATRIX& D)
		{
			if (H.data == NULL || H.m <= 0 || H.n <= 0 || D.m <= 0 || D.n <= 0)
			{
				std::cout << "Invalid Matrix! " << __LINE__ << std::endl;
				return false;
			}
			if (cudaMemcpy(H.data, D.data, H.m * H.n * sizeof(float), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
			{
				std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ << std::endl;
				return false;
			}
			return true;
		}
}
	/****************************************
	routine that executes on host, prepares and calls device functions
	@return N/A
	@return N/A
	****************************************/
	void callMatrixSummation()
	{
		std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
		std::cout << "call Matrix Summation .. " << std::endl;
		std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
		//definitions
		const int M = 20;
		const int N = 20;
		const int blockSize = 16;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float time;
		dim3 dimBlock(blockSize, blockSize,1);
		dim3 dimGrid(N / blockSize + (N % blockSize == 0 ? 0 : 1), M / blockSize + (M % blockSize == 0 ? 0 : 1), 1);
		matrix::MATRIX X_h, Y_h, X_d, Y_d;
		//allocations
		if (matrix::allocateMatrixForDevice(M, N, X_d) && allocateMatrixForDevice(M, N, Y_d) && allocateMatrixForHost(M, N, X_h) && allocateMatrixForHost(M, N, Y_h))
		{
			fillMatrix(X_h);
			fillMatrix(Y_h);
			std::cout << std::fixed<<"Matrix last number:" << Y_h.data[M*N - 1] << " and " << X_h.data[M*N - 1] << std::endl;
			
			copyHostToDevice(X_d, X_h);
			copyHostToDevice(Y_d, Y_h);

			cudaEventRecord(start, 0);
			matrix::matrix_summation << <dimGrid, dimBlock >> >(X_d, Y_d, M, N);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);

			std::cout << cudaGetErrorString(cudaGetLastError()) << " at line:" << __LINE__ << std::endl;
			matrix::copyDeviceToHost(X_h, X_d);
			matrix::copyDeviceToHost(Y_h, Y_d);

			printf("Time for the kernel: %f ms\n", time);
			std::cout << cudaGetErrorString(cudaGetLastError()) << " at line:" << __LINE__ << std::endl;
			matrix::dumpMatrix(X_h);

		}
		// cleanup
		if (X_h.data != NULL)
		{
			delete[] X_h.data;
		}
		if (Y_h.data != NULL)
		{
			delete[] Y_h.data;
		}
		//TODO CUDA null check var mi??
		cudaFree(X_d.data);
		cudaFree(Y_d.data);
	}
	
#define ROW_SIZE 128
	__global__ void matrix_vector_product(int* Mtx, int* Vec, int* VecOut, int i, int j)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		__shared__ int tempMatrix[ROW_SIZE];
		tempMatrix[threadIdx.x] = Vec[threadIdx.x] * Mtx[idx];
		__syncthreads();
		if (threadIdx.x == 0)
		{
			int sum = 0;
			for (int it = 0; it < i; ++it)
			{
				//sum += Mtx[i * blockIdx.x + it];
				sum += tempMatrix[it];
			}
			VecOut[blockIdx.x] = sum;
		}
	}
	
	/****************************************
	routine that executes on host, prepares and calls device functions
	@return N/A
	@return N/A
	****************************************/
	void callMatrixVectorProduct()
	{
		std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
		std::cout << "call Matrix Vector Product .. " << std::endl;
		std::cout << "::::::::::::::::::::::::::::::::" << std::endl;
		//definitions
		const int i = ROW_SIZE;
		const int j = ROW_SIZE;
		const int blockSize = i;

		float time;

		dim3 dimBlock(blockSize, 1);
		dim3 dimGrid(j, 1);

		int* Mtx_h, *Mtx_d;
		int* Vec_h, *Vec_d, *VecOut_d;

		// memory allocations
		int sizeOfVector = j * sizeof(int);
		
		Vec_h = (int*)malloc(sizeOfVector);
		Mtx_h = (int*)malloc(i*sizeOfVector);
		
		if (cudaMalloc((void**)&Vec_d, sizeOfVector) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ - 1 << std::endl;
		}

		if (cudaMalloc((void**)&VecOut_d, sizeOfVector) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ - 1 << std::endl;
		}

		if (cudaMalloc((void **)&Mtx_d, i * sizeOfVector) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ -1 << std::endl;
		}

		// filling host memory

		for (int it = 0; it < j  ; ++it)
		{
			Vec_h[it] = it;
		}

		for (int it = 0; it < j * i ; ++it)
		{
			Mtx_h[it] = 1;
		}
		//std::cout << "=============" << std::endl;
		//for (int it = 0; it < i * j; ++it)
		//{
		//	if (fmod((double)it, (double)(j)) == 0)
		//		std::cout << "\\n" << std::endl;
		//	std::cout << Mtx_h[it] << " ";

		//}
		//std::cout << "\\n" << std::endl;
		//std::cout << "=============" << std::endl;

		std::cout << "==== Vector ====" << std::endl;
		for (int it = 0; it < j; ++it)
		{
			std::cout << Vec_h[it] << " ";
		}

		if(cudaMemcpy(Vec_d, Vec_h, sizeOfVector, cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ - 1 << std::endl;
		}
		if (cudaMemcpy(Mtx_d, Mtx_h, sizeOfVector * i, cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ - 1 << std::endl;
		}

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	
		//////////////////////////////////////////////////////////////////////////////////
		matrix_vector_product << < dimGrid, dimBlock>> >(Mtx_d, Vec_d, VecOut_d,i, j);
		//////////////////////////////////////////////////////////////////////////////////

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("\nTime for the kernel: %f ms\n", time);

		std::cout << cudaGetErrorString(cudaGetLastError()) << " at line:" << __LINE__ << std::endl;

		std::cout << cudaGetErrorString(cudaGetLastError()) << " at line:" << __LINE__ -1 << std::endl;
		
		if (cudaMemcpy(Vec_h, VecOut_d, sizeOfVector, cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ - 1 << std::endl;
		}
		if (cudaMemcpy(Mtx_h, Mtx_d, sizeOfVector * i, cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << " " << __LINE__ - 1 << std::endl;
		}
		
			

		


		std::cout << "==== Vector Result====" << std::endl;
		for (int it = 0; it < j; ++it)
		{
			std::cout <<Vec_h[it] << " ";
		}
		std::cout << "\\n" << std::endl;
		std::cout << "=============" << std::endl;

		//for (int it = 0; it < i * j; ++it)
		//{
		//	if (fmod((double)it, (double)(j)) == 0)
		//		std::cout << "\\n" << std::endl;
		//	std::cout << Mtx_h[it] << " ";

		//}
		//std::cout << "\\n" << std::endl;
		//std::cout << "=============" << std::endl;
		// cleanup
		if (Mtx_h != NULL)
		{
			delete[] Mtx_h;
		}
		if (Vec_h != NULL)
		{
			delete[] Vec_h;
		}
		//TODO CUDA null check var mi??
		cudaFree(Mtx_d);
		cudaFree(Vec_d);
	}


// main routine that executes on the host
int main(void)
{
	callInnerProduct();
	callMatrixSummation();
	callMatrixVectorProduct();
	getchar();
	return 0;
}
