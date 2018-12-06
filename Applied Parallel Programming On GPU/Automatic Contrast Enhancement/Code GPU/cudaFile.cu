#include "cudaFile.cuh"
#include <iostream>
__global__ void subtractPixels(Npp8u* pSrc_Dev, Npp8u* pDest_Dev,unsigned int size,Npp8u min)
{
	unsigned int tid = threadIdx.x; //thread id
	unsigned int gid = (blockDim.x * blockIdx.x) + tid; //global id
	if(gid >= size) 
	{
		return;
	}
	Npp8u val = pSrc_Dev[gid];
	val = val - min;
	pDest_Dev[gid] = val;
}

__global__ void multAndDivPixels(Npp8u* pSrc_Dev,unsigned int size,Npp8u min, Npp8u nConstant, Npp8u nPower)
{
	unsigned int tid = threadIdx.x; //thread id
	unsigned int gid = (blockDim.x * blockIdx.x) + tid; //global id
	if(gid >= size) 
	{
		return;
	}
	 Npp8u val = pSrc_Dev[gid] * nConstant;
	pSrc_Dev[gid] = val / 7;
}

void callSubtract(Npp8u* pSrc_Dev, Npp8u* pDest_Dev, unsigned int size,Npp8u min, int gridSize, int blockSize)
{
	subtractPixels <<<gridSize, blockSize >>>(pSrc_Dev, pDest_Dev, size, min);
	std::cout<<__FUNCTION__<<" :: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
}

void callMultAndDivPixels(Npp8u* pSrc_Dev, unsigned int size,Npp8u min,  Npp8u nConstant, Npp8u nPower, int gridSize, int blockSize)
{
	multAndDivPixels <<<gridSize, blockSize >>>(pSrc_Dev, size, min,  nConstant,  nPower);
	std::cout<<__FUNCTION__<<" :: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
}