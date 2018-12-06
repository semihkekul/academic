#include "stdafx.h"
#include <stdio.h>

#include <cuda.h>
#include <cutil.h>

#define __DEBUG__ printf("Function %s() line %d\n",__FUNCTION__,__LINE__);
#define NA -999 
#define MATRIX_DIM 500
#define DATA_SIZE 1996
#define __DEVICE__ 
namespace COO
	{
	__global__ void GPU_kernel( 
		float * data ,
		int * rowA ,
		int * colA ,
		int * rowB ,
		int * colB ,
		float* Z
		)
		{
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if(idx >= MATRIX_DIM) return;
		
		for(int y = 0; y< MATRIX_DIM; y++)
			{
			int offsetA;
			for(int k=0; k< DATA_SIZE; k++)
				{
				if (rowA[k]==idx) 
					{
					offsetA=k;
					break;
					}
				}

			float dot = 0;
			for(int i =offsetA; rowA[i] == idx;i++ )
				{


				for(int a = 0 ; a<DATA_SIZE ;a++ )
					{
					dot = 0;
					if(colA[i] == rowB[a] )
						{
						if(colB[a]==y)
							{
							dot+= data[i]*data[a];
							}
						}
						Z[MATRIX_DIM*idx+y] += dot;
					}
				}
			}

		}
	__host__ void CPU_kernel ( 
		float * data ,
		int * rowA ,
		int * colA ,
		int * rowB ,
		int * colB ,
		int idx,
		float* Z
		)
		{
		if(idx >= MATRIX_DIM) return;
		for(int y = 0; y< MATRIX_DIM; y++)
			{
			int offsetA;
			for(int k=0; k< DATA_SIZE; k++)
				{
				if (rowA[k]==idx) 
					{
					offsetA=k;
					break;
					}
				}

			float dot = 0;
			for(int i =offsetA; rowA[i] == idx;i++ )
				{


				for(int a = 0 ; a<DATA_SIZE ;a++ )
					{
					dot = 0;
					if(colA[i] == rowB[a] )
						{
						if(colB[a]==y)
							{
							dot+= data[i]*data[a];
							}
						}
					Z[MATRIX_DIM*y+idx]+=dot;
					}



				}
			}

		}
	}
int main(void)
	{
	printf("\nAssignment -2- : COO Matrix  Multiplication\n");
	FILE*  file;
	unsigned int cudatTimer;

	int* row_h = (int*)malloc(sizeof(int)*DATA_SIZE);
	int* col_h = (int*)malloc(sizeof(int)*DATA_SIZE);
	float* data_h = (float*)malloc(sizeof(float)*DATA_SIZE);
	float* Z_h = (float*)malloc(sizeof(float)*MATRIX_DIM*MATRIX_DIM); 
	for(int i = 0; i < MATRIX_DIM* MATRIX_DIM; i++)
		{
			Z_h[i] = .0f;
		}

	int* row_d;
	int* col_d;
	float* data_d;
	float* Z_d;

	file = fopen("C:\\gate713\\coo500\\coo500d.txt","r");
	if (file==NULL) perror("file");
	int iter;
	for(iter = 0; iter < DATA_SIZE; iter++)
		{
		fscanf(file,"%f",&data_h[iter]);
		}
	printf("%d\n ",iter);
	fclose(file);

	__DEBUG__
		file = fopen("C:\\gate713\\coo500\\coo500r.txt","r");
	if (file==NULL) perror("file");
	for( iter = 0; iter < DATA_SIZE; iter++)
		{
		fscanf(file,"%d",&row_h[iter]);
		}
	fclose(file);
	printf("%d\n ",iter);
	__DEBUG__
		file = fopen("C:\\gate713\\coo500\\coo500c.txt","r");
	if (file==NULL) perror("file");
	for( iter = 0; iter < DATA_SIZE; iter++)
		{
		fscanf(file,"%d",&col_h[iter]);
		}
	fclose(file);
	printf("%d\n ",iter);
	
#ifndef __DEVICE__
	__DEBUG__
		cudatTimer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&cudatTimer));
	CUT_SAFE_CALL(cutStartTimer(cudatTimer));
	for(int idx=0;idx<MATRIX_DIM; idx++)
		{
		COO::CPU_kernel(data_h,row_h,col_h,row_h,col_h,idx,Z_h);
		}
	CUT_SAFE_CALL(cutStopTimer(cudatTimer));
	printf( "Processing time on host: %f (ms)\n", cutGetTimerValue(cudatTimer));
#endif
	
#ifdef __DEVICE__
	

	cudaMalloc((void**)&row_d, sizeof(int)* DATA_SIZE);

	CUDA_SAFE_CALL(cudaMalloc((void**)&col_d,sizeof(int)* DATA_SIZE));

	CUDA_SAFE_CALL(cudaMalloc((void**)&data_d,sizeof(float)* DATA_SIZE));

	CUDA_SAFE_CALL(cudaMalloc((void**)&Z_d,sizeof(float)* MATRIX_DIM * MATRIX_DIM));
	cudatTimer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&cudatTimer));
	CUT_SAFE_CALL(cutStartTimer(cudatTimer));
	cudaMemcpy(row_d,row_h,sizeof(int)* DATA_SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(col_d,col_h,sizeof(int)* DATA_SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(data_d,data_h,sizeof(float)* DATA_SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(Z_d,Z_h,sizeof(float)* MATRIX_DIM*MATRIX_DIM,cudaMemcpyHostToDevice);
	dim3 dimGrid(200,1,1);
	dim3 dimBlock(250,1,1);

	COO::GPU_kernel<<<dimGrid,dimBlock>>>(data_d,row_d,col_d,row_d,col_d,Z_d);
	CUT_SAFE_CALL(cutStopTimer(cudatTimer));
	printf( "Processing time on device: %f (ms)\n", cutGetTimerValue(cudatTimer));
	CUT_SAFE_CALL(cutDeleteTimer(cudatTimer));
	cudaMemcpy(Z_h,Z_d,sizeof(float)* MATRIX_DIM*MATRIX_DIM,cudaMemcpyDeviceToHost);
	cudaFree(row_d);
	cudaFree(col_d);
	cudaFree(data_d);
	cudaFree(Z_d);
#endif
	free(row_h);
	free(col_h);
	free(data_h);
	free(Z_h);
	
	
	return 0;

	}
