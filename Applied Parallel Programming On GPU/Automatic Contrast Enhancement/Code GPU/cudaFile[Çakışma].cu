#include "cudaFile.cuh"
__global__ void subtractPixels(Npp8u* pSrc_Dev, Npp8u* pDest_Dev,unsigned int size,Npp8u min)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x; //thread id
	unsigned int gid = (blockDim.x * blockIdx.x) + tid; //global id
	if(gid >= size) 
	{
		return;
	}
	int val = (int)pSrc_Dev[gid];
	val = (int)val - (int)min;
	pDest_Dev[gid] = (Npp8u)val;
}

void callSubtract(Npp8u* pSrc_Dev, Npp8u* pDest_Dev, unsigned int size,Npp8u min, int gridSize, int blockSize)
{
	subtractPixels <<<gridSize, blockSize >>>(pSrc_Dev, pDest_Dev, size, min);
}