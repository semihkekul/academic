/*
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// This example implements the contrast adjustment on an 8u one-channel image by using
// Nvidia Performance Primitives (NPP). 
// Assume pSrc(i,j) is the pixel value of the input image, nMin and nMax are the minimal and 
// maximal values of the input image. The adjusted image pDst(i,j) is computed via the formula:
// pDst(i,j) = (pSrc(i,j) - nMin) / (nMax - nMin) * 255 
//
// The code flow includes five steps:
// 1) Load the input image into the host array;
// 2) Allocate the memory space on the GPU and copy data from the host to GPU;
// 3) Call NPP functions to adjust the contrast;
// 4) Read data back from GPU to the host;
// 5) Output the result image and clean up the memory.

#include <iostream>
#include <fstream>
#include <sstream>

#include "cudaFile.cuh"
void multAndDivPixels(Npp8u * data, int size, Npp8u nConstant, int nScaleFactor, int nPower)
{
	for(int i = 0; i < size; ++i)
	{
		int val = data[i] * (int)nConstant;
		data[i] = (Npp8u)(val / nPower);
	}
}

int findMin(Npp8u * data, int size)
{
	int min = 255;
	for(int i = 0; i < size; ++i)
	{
		if(data[i] < min) min = data[i];
	}
	return min;
}

int findMax(Npp8u * data, int size)
{
	int max = 0;
	for(int i = 0; i < size; ++i)
	{
		if(data[i] > max) max = data[i];
	}
	return max;
}

void subtractPixels(Npp8u * data, int size, Npp8u min)
{
	for(int i = 0; i < size; ++i)
	{
		data[i] -= min;
	}
}


// Function declarations.
Npp8u *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);

void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray);

// Main function.
int 
main(int argc, char ** argv)
{
		// Host parameter declarations.	
	Npp8u * pSrc_Host, * pDst_Host;
	int   nWidth, nHeight, nMaxGray;     
	
		// Load image to the host.
    std::cout << "Load PGM file." << std::endl;
	pSrc_Host = LoadPGM("lena_before.pgm", nWidth, nHeight, nMaxGray);
	pDst_Host = new Npp8u[nWidth * nHeight];
  
		// Device parameter declarations.
    Npp8u	 * pSrc_Dev, * pDst_Dev;	
	Npp8u    * pMin_Dev, * pMax_Dev;	
	Npp8u    * pBuffer_Dev;				
	Npp8u    nMin_Host, nMax_Host;	    
	NppiSize oROI;						
    int		 nSrcStep_Dev, nDstStep_Dev;
	int		 nBufferSize_Host = 0;	

		// Copy the image from the host to GPU
    oROI.width = nWidth;
    oROI.height = nHeight;
	pSrc_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nSrcStep_Dev);  
	pDst_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nDstStep_Dev);  
    std::cout << "Copy image from host to device." << std::endl;
	//cudaMemcpy2D(pSrc_Dev, nSrcStep_Dev, pSrc_Host, nWidth, nWidth, nHeight, cudaMemcpyHostToDevice);
	cudaMemcpy(pSrc_Dev, pSrc_Host, nWidth * nHeight * sizeof(Npp8u), cudaMemcpyHostToDevice) ;
	std::cout << "Process the image on GPU." << std::endl;
		// Allocate device buffer for the MinMax primitive -- this is only necessary for nppi, we can simply return into nMin_Host and n_Max_Host
    cudaMalloc(reinterpret_cast<void **>(&pMin_Dev), sizeof(Npp8u)); // You won't need these lines
	cudaMalloc(reinterpret_cast<void **>(&pMax_Dev), sizeof(Npp8u)); // You won't need these lines
	//nppiMinMaxGetBufferHostSize_8u_C1R(oROI, &nBufferSize_Host);  // You won't need these lines 
    nMin_Host = findMin(pSrc_Host, nWidth * nHeight);
	nMax_Host = findMax(pSrc_Host, nWidth * nHeight);
	cudaMalloc(reinterpret_cast<void **>(&pBuffer_Dev), nBufferSize_Host); // You won't need these lines
		
	// REPLACE THIS PART WITH YOUR KERNELs
	// Compute the min and the max.
	//nppiMinMax_8u_C1R(pSrc_Dev, nSrcStep_Dev, oROI, pMin_Dev, pMax_Dev, pBuffer_Dev); // // Replace this line with your KERNEL1 call (KERNEL1: your kernel calculating the minimum and maximum values and returning them here)
	//cudaMemcpy(&nMin_Host, pMin_Dev, sizeof(Npp8u), cudaMemcpyDeviceToHost); // You won't need these lines to get the min and max. Return nMin_Host from your kernel function 
	//cudaMemcpy(&nMax_Host, pMax_Dev, sizeof(Npp8u), cudaMemcpyDeviceToHost); // You won't need these lines to get the min and max. Return nMax_Host from your kernel function
        
        // Call SubC primitive.
	callSubtract(pSrc_Dev, pDst_Dev, nWidth* nHeight, nMin_Host, 512, 512); //KERNEL2: your kernel subtracting the nMin_Host from all the pixels)
	
        // Compute the optimal nConstant and nScaleFactor for integer operation see GTC 2013 Lab NPP.pptx for explanation
    int nScaleFactor = 0; 
    int nPower = 1; 
    while(nPower * 255.0f / (nMax_Host - nMin_Host) < 255.0f) 
    { 
        nScaleFactor ++;
        nPower *= 2;
    }
    Npp8u nConstant = static_cast<Npp8u>(255.0f / (nMax_Host - nMin_Host) * (nPower / 2)); //you won't need these calculations
	
	   // Call MulC primitive.
	//cudaMemcpy2D(pDst_Host, nWidth*sizeof(Npp8u), pDst_Dev, nDstStep_Dev, nWidth*sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
	
    //nppiMulC_8u_C1IRSfs(nConstant, pDst_Dev, nDstStep_Dev, oROI, nScaleFactor -1); // Replace this line with your KERNEL3 call (KERNEL3: your kernel multiplying all the pixels with the nConstant and then dividing them by nScaleFactor -1 to achieve: 255/(nMax_Host-nMinHost)))
	
	callMultAndDivPixels(pDst_Dev,nWidth*nHeight,nConstant,nScaleFactor-1 , nPower / 2, 512, 512);
	cudaMemcpy(pDst_Host, pDst_Dev, nWidth * nHeight * sizeof(Npp8u), cudaMemcpyDeviceToHost);
	//-------------------
		// Copy result back to the host.
    std::cout << "Work done! Copy the result back to host." << std::endl;
  //  cudaMemcpy2D(pDst_Host, nWidth*sizeof(Npp8u), pDst_Dev, nDstStep_Dev, nWidth*sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
		
		// Output the result image.
    std::cout << "Output the PGM file." << std::endl;
    WritePGM("lena_after.pgm", pDst_Host, nWidth, nHeight, nMaxGray);

		// Clean up.
    std::cout << "Clean up." << std::endl;
	delete[] pSrc_Host;
	delete[] pDst_Host;	

	nppiFree(pSrc_Dev);
	nppiFree(pDst_Dev);	
	cudaFree(pBuffer_Dev);
	nppiFree(pMin_Dev);
	nppiFree(pMax_Dev);
	getchar();
    return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )

// Load PGM file.
Npp8u *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{	
    char aLine[256];
	FILE * fInput = fopen(sFileName, "r");
    if(fInput == 0)
    {
        perror("Cannot open file to read");
        exit(EXIT_FAILURE);
    }
    // First line: version
    fgets(aLine, 256, fInput);
    std::cout << "\tVersion: " << aLine;
    // Second line: comment
    fgets(aLine, 256, fInput);
    std::cout << "\tComment: " << aLine;
    fseek(fInput, -1, SEEK_CUR);
    // Third line: size
    fscanf(fInput, "%d", &nWidth);
    std::cout << "\tWidth: " << nWidth;
    fscanf(fInput, "%d", &nHeight);
    std::cout << " Height: " << nHeight << std::endl;
    // Fourth line: max value
    fscanf(fInput, "%d", &nMaxGray);
    std::cout << "\tMax value: " << nMaxGray << std::endl;
    while(getc(fInput) != '\n');
	// Following lines: data
	Npp8u * pSrc_Host = new Npp8u[nWidth * nHeight];	
    for (int i = 0; i < nHeight; ++ i)
        for (int j = 0; j < nWidth; ++ j)
			pSrc_Host[i*nWidth+j] = fgetc(fInput);
	fclose(fInput);

	return pSrc_Host;
}

// Write PGM image.
void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray)
{
    FILE * fOutput = fopen(sFileName, "w+");
    if(fOutput == 0)
    {
        perror("Cannot open file to read");
        exit(EXIT_FAILURE);
    }
    char * aComment = "# Created by NPP";
    fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
	for (int i = 0; i < nHeight; ++ i)
		for(int j = 0; j < nWidth; ++ j)
            fputc(pDst_Host[i*nWidth+j], fOutput);
    fclose(fOutput);
}

