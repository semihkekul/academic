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
#include "npp.h"


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

//************************************
// Method:    subtractPixels subtracting the nMin_Host from all the pixels
// FullName:  subtractPixels
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: Npp8u * data
// Parameter: int size
// Parameter: Npp8u min
//************************************
void subtractPixels(Npp8u * data, int size, Npp8u min)
{
	for(int i = 0; i < size; ++i)
	{
		data[i] -= min;
	}
}

//************************************
// Method:    multAndDivPixels multiplying all the pixels with the nConstant and then dividing them by nScaleFactor -1 to achieve: 255/(nMax_Host-nMinHost)))
// FullName:  multAndDivPixels
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: Npp8u * data
// Parameter: int size
// Parameter: Npp8u nConstant
// Parameter: int nScaleFactor
//************************************
void multAndDivPixels(Npp8u * data, int size, Npp8u nConstant, int nScaleFactor, int nPower)
{
	for(int i = 0; i < size; ++i)
	{
		int val = data[i] * (int)nConstant;
		data[i] = (Npp8u)(val / nPower);
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
	Npp8u    nMin_Host, nMax_Host;	   					
	int		 nBufferSize_Host = 0;	
	std::cout << "Process the image on CPU." << std::endl;

	nMin_Host = findMin(pSrc_Host, nWidth * nHeight);
	nMax_Host = findMax(pSrc_Host, nWidth * nHeight);
	std::cout << "Min: "<<(int)nMin_Host <<" Max: "<<(int)nMax_Host<<std::endl;    
	subtractPixels(pSrc_Host, nWidth * nHeight, nMin_Host);
	// Compute the optimal nConstant and nScaleFactor for integer operation see GTC 2013 Lab NPP.pptx for explanation
	int nScaleFactor = 0; 
	int nPower = 1; 
	while(nPower * 255.0f / (nMax_Host - nMin_Host) < 255.0f) 
	{ 
		nScaleFactor ++;
		nPower *= 2;
	}
	Npp8u nConstant = static_cast<Npp8u>(255.0f / (nMax_Host - nMin_Host) * (nPower / 2)); //you won't need these calculations
	std::cout << "Constant: "<<(int)nConstant<<", Scale factor: " <<nScaleFactor<<", Power:" <<nPower<< std::endl;
	// Call MulC primitive
	multAndDivPixels(pSrc_Host,nWidth*nHeight,nConstant,nScaleFactor-1 , nPower / 2);
	// Output the result image.
	std::cout << "Output the PGM file." << std::endl;
	WritePGM("lena_after.pgm", pSrc_Host, nWidth, nHeight, nMaxGray);
	// Clean up.
	std::cout << "Clean up." << std::endl;
	delete[] pSrc_Host;
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

