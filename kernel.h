#pragma once
#ifndef kernel
#define kernel

#include <iostream>
#include "complex.h"
#include "vector2.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void mandelbrotKernel(int maxIt, int length, int width, int height, double xMax, double xMin, double yMax, double yMin, double xOffset, double yOffset, double xFix, double yFix, double zoom, int *vec)
{
    int stride = blockDim.x * gridDim.x;
	int loop;
    double c, imZ, rlZ, imC, rlC;
    
	double xzoom = ((xMax - xMin) / width);
	double yzoom = ((yMax - yMin) / height);

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length; i += stride)
	{
		rlC = xzoom * ((i / height) + xOffset + (xFix * (zoom * 4))) - abs(xMin);
		imC = yzoom * ((i % height) + yOffset + (yFix * (zoom * 4))) - abs(yMin);

		c = 0;
		imZ = 0;
		rlZ = 0;
		loop = 0;

		while (loop < maxIt && c <= 4)
		{
			loop++;

			Pow(&imZ, &rlZ);
			Add(&imZ, &rlZ, imC, rlC);
			Norm(&c, imZ, rlZ);
		}

		vec[i] = loop;
	}
}

int* cut(int *vec, Vector2 size) 
{
	int n = size.X * size.Y;
	int *hvec = (int*)malloc(n * sizeof(int));

	int delta = size.X - size.Y;
	int skips = 0;
	for (int i = 0; i < n; i++) 
	{
		hvec[i] = vec[i + skips * delta];
		if (i+1 % (int)size.Y == 0) 
		{
			 i += delta;
			 skips++;
		}
	}

	free(vec);

	return hvec;
}

int* mandelbrot(int iterations, double zoom, Vector2 size, Vector2 center, Vector2 fix)
{
    int n = size.X * size.X;

    int *hvec = (int*)malloc(n * sizeof(int));
    int *dvec = 0;
    cudaMalloc((void**)&dvec, n * sizeof(int));
    
	mandelbrotKernel<<<16, 1024>>>(iterations, n, size.X, size.X, 2 * (1 / zoom), -2 * (1 / zoom), 2 * (1 / zoom), -2 * (1 / zoom), center.X, center.Y, fix.X, fix.Y, zoom, dvec);

	cudaMemcpy(hvec, dvec, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dvec);

	return cut(hvec, size);
}

#endif