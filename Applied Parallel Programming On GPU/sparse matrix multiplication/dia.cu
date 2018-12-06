#include "stdafx.h"
#include <stdio.h>

#include <cuda.h>
#include <cutil.h>
#define NA -999.0f 
namespace DIA
	{

	__host__ void CPU_kernel ( const  int num_rows ,
		const  int num_cols ,
		const  int num_diags ,
		int   * offsets ,
		float * data ,
		float * x ,
		float * y ,
		int idx,
		int idy)
		{

		if ( idx < num_rows * num_cols){  // be careful here
			float dot = 0;
			int n = 0;
			for ( n = 0; n < num_diags ; n ++){
				int r = idx + offsets [ n ];
				float valA = data[num_cols*idx+n];
				if ( r >= 0 && r < num_rows )
					{
					int magicNumber = idy-r;
					for(int b=0; b < num_diags/*for offB*/; b++)
						{

						if (offsets[b]==magicNumber)
							{
							float valB = data[ b + r*num_cols ];
							dot += valA * valB;
							}
						}
	
					}
				}
			y [ idx ] += dot ;
			}
		}

	__global__ void
		GPU_kernel (  int num_rows ,
		int num_cols ,
		int num_diags ,
		int dimMatrix,
		int * offA ,
		int * offB ,
		float * dataA,
		float * dataB,
		float * C)
		{
		int idxX = threadIdx.x + blockDim.x*blockIdx.x;
		int idxY = threadIdx.y + blockDim.y*blockIdx.y;
		if ( idxX < num_rows && idxY < num_rows){  
			float dot = 0;
			int n = 0;
			for ( n = 0; n < num_diags ; n ++){
				int r = idxX + offA [ n ];
				float valA = dataB[num_cols*idxX+n];
				if ( r >= 0 && r < num_rows )
					{
					int magicNumber = idxY-r;

					for(int b=0; b < num_diags/*for offB*/; b++)
						{

						if (offB[b]==magicNumber)
							{
							float valB = dataB[ b + r*num_cols ];
							dot += valA * valB;
							}
						}


					}
				}
			C[ idxX*dimMatrix + idxY] += dot ;

			}
		}
	__global__ void
		GPU_kernel_texture (  int num_rows ,
		int num_cols ,
		int num_diags ,
		int dimMatrix,
		int * offA ,
		int * offB ,
		float * dataA,
		float * dataB,
		float * C)
		{
		int idxX = threadIdx.x + blockDim.x*blockIdx.x;
		int idxY = threadIdx.y + blockDim.y*blockIdx.y;
		if ( idxX < num_rows && idxY < num_rows){  
			float dot = 0;
			int n = 0;
			for ( n = 0; n < num_diags ; n ++){
				int r = idxX + offA [ n ];
				float valA = dataB[num_cols*idxX+n];
				if ( r >= 0 && r < num_rows )
					{
					int magicNumber = idxY-r;

					for(int b=0; b < num_diags/*for offB*/; b++)
						{

						if (offB[b]==magicNumber)
							{
							float valB = dataB[ b + r*num_cols ];
							dot += valA * valB;
							}
						}


					}
				}
			C[ idxX*dimMatrix + idxY] += dot ;

			}
		}
	}

int main(void)
	{
	printf("\nAssignment -2- : Sparse Matrix  Multiplication\n");
	float dataA[]={NA,NA,5.0f,6.0f,1.0f,2.0f,3.0f,4.0f,7.0f,8.0f,9.0f,NA};
	float dataB[]={NA,1.0f,7.0f,NA,2.0f,8.0f,5.0f,3.0f,9.0f,6.0f,4.0f,NA};
	float* dataA_h=dataA;
	float* dataB_h=dataB;
	int offsets[]={-2,0,1};
	int* offsetsA_h=offsets;
	int* offsetsB_h=offsets;
	float* C_h;


	float* C_d;
	float* dataA_d,*dataB_d;
	int * offsetsA_d,*offsetsB_d;


	unsigned int timer;
	/** allocate memory for host **/

	C_h = (float *)malloc(sizeof(float)*16);   

	/** allocate memory for device **/
	cudaMalloc((void **) &dataA_d, sizeof(float)*12);   
	cudaMalloc((void **) &dataB_d, sizeof(float)*12);   
	cudaMalloc((void **) &offsetsA_d, sizeof(float)*4);   
	cudaMalloc((void **) &offsetsB_d, sizeof(float)*4);  
	cudaMalloc((void **) &C_d, sizeof(float)*16);   



	/** timer start **/
	timer = 0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	/** copy matrix values to device **/
	cudaMemcpy(dataA_d, dataA_h, 12*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dataB_d, dataB_h, 12*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(offsetsA_d, offsetsA_h, 4*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(offsetsB_d, offsetsA_h, 4*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(C_d, C_h, 16*sizeof(float), cudaMemcpyHostToDevice);
	dim3  grid( 100, 1, 1);
	dim3  block( 10, 20, 1);
	/** loop through each entry of output matrix**/
    DIA::GPU_kernel<<<grid,block>>>(4,3,3,4,offsetsA_d,offsetsB_d,dataA_d,dataB_d,C_d);
	/** copy output matrix value to host **/
	cudaMemcpy(C_h, C_d, 16*sizeof(float), cudaMemcpyDeviceToHost);

	CUT_SAFE_CALL(cutStopTimer(timer));
	printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)); 

	for(int i=0;i<4;i++)
		{
		printf("%f %f %f %f\n ",C_h[i*4],C_h[i*4+1],C_h[i*4+2],C_h[i*4+3]);
		}

	/** cleaning**/
	cudaFree(dataA_d);
	cudaFree(dataB_d);
	cudaFree(offsetsA_d);
	cudaFree(offsetsB_d);
	cudaFree(C_d);


	}
