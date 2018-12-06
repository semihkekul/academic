#include "stdafx.h"
#include <stdio.h>

#include <cuda.h>
#include <cutil.h>
#define NA -999 
namespace CSC
	{

	int linearSearch(int number, int* a, int size)
		{
		for(int i=0; i< size; i++)
			{
			if (a[i]==number) return i;
			}
		return NA;
		}
	__host__ void CPU_kernel ( const  int num_dataA ,
		float * dataA ,
		int * ptrA ,
		int * indicesA ,
		const  int num_dataB ,
		float* dataB,
		int * ptrB ,
		int * indicesB ,
		int rowNo,
		int colNo,
		float* Z
		)
		{
		printf("******************* Row %d Col %d******************** \n",rowNo,colNo);
		for(int indiceIterA = 0; indiceIterA < num_dataA; indiceIterA++ )
			{
			if(indicesA[indiceIterA] == rowNo) //row finded
				{
					int dotColA = -1;
					for(int search = 1;    search < 5 /*ptr_sizeB*/; search++)
						{
						if( indiceIterA < ptrA[search] && indiceIterA >= ptrB[search-1])
							{
							dotColA = search - 1; /*should be equal to dotRowB*/
							break;
							}
						}
				}
			}
		}
	}
int main(void)
	{
	printf("\nAssignment -2- : CSC Matrix  Multiplication\n");
	int ptr[] =		{0,2,3,6};
	int indices[] = {0,2,2,0,1,2};
	float data[] =	{1,2,3,4,5,6};

	float Z[9]={0};
	for(int idx=0;idx<3; idx++)
		{
		//for(int idy=0;idy<3; idy++)
			CSC::CPU_kernel(6, data,ptr,indices,6,data,ptr,indices,idx,idy,Z);

		}
	for(int i=0;i<3;i++)
		{
		printf("%f %f %f %f\n ",Z[i*4],Z[i*4+1],Z[i*4+2],Z[i*4+3]);
		}



	return 0;

	}
