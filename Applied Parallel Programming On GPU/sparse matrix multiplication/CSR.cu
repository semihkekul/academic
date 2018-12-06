#include "stdafx.h"
#include <stdio.h>

#include <cuda.h>
#include <cutil.h>
#define NA -999 
namespace CSR
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

		for(int indiceIterA = ptrA[rowNo]; indiceIterA < ptrA[rowNo+1] ; indiceIterA++)
			{

			printf("\nindiceIterA is %d\n",indiceIterA);
			for(int indiceIterB = 0; indiceIterB < num_dataB; indiceIterB++)
				{
				if(indicesB[indiceIterB] == colNo)
					{
					int rowB = -1;
					for(int search = 1;    search < 5 /*ptr_sizeB*/; search++)
						{
						//	printf("search %d\n",search);
						//	fflush(stdout);
						if( indiceIterB < ptrB[search] && indiceIterB >= ptrB[search-1])
							{
							rowB = search - 1;
							break;
							}
						}

					printf("rowB is %d for iidiceIterB %d\n",rowB,indiceIterB);
					if(rowB == indicesA[indiceIterA])
						{
						Z[rowNo * 4 + colNo]+= dataA[indiceIterA] * dataB[indiceIterB];
						printf("%f * %f \n",dataA[indiceIterA], dataB[indiceIterB]);
						}
					}
				}
			}

		}
	}
int main(void)
	{
	printf("\nAssignment -2- : CSR Matrix  Multiplication\n");
	int ptr[]		= {0 ,2 ,4 ,7 ,9};
	int indices[]	= {0, 1, 1, 2, 0, 2, 3, 1, 3};
	float data[]	= {1, 7, 2, 8, 5, 3, 9, 6, 4};
	float Z[16]={0};
	for(int idx=0;idx<4; idx++)
		{
		for(int idy=0;idy<4; idy++)
			CSR::CPU_kernel(9, data,ptr,indices,9,data,ptr,indices,idx,idy,Z);

		}
	for(int i=0;i<4;i++)
		{
		printf("%f %f %f %f\n ",Z[i*4],Z[i*4+1],Z[i*4+2],Z[i*4+3]);
		}



	return 0;

	}
