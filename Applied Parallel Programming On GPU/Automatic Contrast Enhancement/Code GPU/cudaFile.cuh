#include <cuda.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
void callSubtract(Npp8u* pSrc_Dev, Npp8u* pDest_Dev, unsigned int size,Npp8u min, int gridSize, int blockSize);
void callMultAndDivPixels(Npp8u* pSrc_Dev,unsigned int size,Npp8u min, Npp8u nConstant, Npp8u nPower, int gridSize, int blockSize);