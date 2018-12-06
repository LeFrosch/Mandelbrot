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

__global__ void mandelbrotKernel(int maxIt, int length, int width, int height, double xMax, double xMin, double yMax, double yMin, double xOffset, double yOffset, double xFix, double yFix, double zoom, double *vec)
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

__global__ void mandelbrotKernelsmooth(int maxIt, int length, int width, int height, double xMax, double xMin, double yMax, double yMin, double xOffset, double yOffset, double xFix, double yFix, double zoom, double *vec, double smoothVal)
{
    int stride = blockDim.x * gridDim.x;
	int loop;
    double c, imZ, rlZ, imC, rlC;
    double p;

	double xzoom = ((xMax - xMin) / width);
	double yzoom = ((yMax - yMin) / height);

	double ab = 0;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length; i += stride)
	{
		rlC = xzoom * ((i / height) + xOffset + (xFix * (zoom * 4))) - abs(xMin);
		imC = yzoom * ((i % height) + yOffset + (yFix * (zoom * 4))) - abs(yMin);

		c = 0;
		imZ = 0;
		rlZ = 0;
		p = 1;
		vec[i] = 0;

		for (loop = 0; loop < maxIt; loop++)
		{
			sqrABS(&ab, imZ, rlZ);

			if (ab > 1000000) 
			{
				vec[i] = log(log(ab) / p) / smoothVal;
				break;
			}

			Pow(&imZ, &rlZ);
			Add(&imZ, &rlZ, imC, rlC);
			Norm(&c, imZ, rlZ);
			p *= 2;
		}
	}
}

double* cut(double *vec, Vector2 size) 
{
	int n = size.X * size.Y;
	double *hvec = (double*)malloc(n * sizeof(double));

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

double* mandelbrot(int iterations, double zoom, Vector2 size, Vector2 center, Vector2 fix, bool smooth, double smoothVal)
{
    int n = size.X * size.X;

    double *hvec = (double*)malloc(n * sizeof(double));
    double *dvec = 0;
    cudaMalloc((void**)&dvec, n * sizeof(double));
	
	if (!smooth)
		mandelbrotKernel<<<16, 1024>>>(iterations, n, size.X, size.X, 2 * (1 / zoom), -2 * (1 / zoom), 2 * (1 / zoom), -2 * (1 / zoom), center.X, center.Y, fix.X, fix.Y, zoom, dvec);
	else 
		mandelbrotKernelsmooth<<<16, 1024>>>(iterations, n, size.X, size.X, 2 * (1 / zoom), -2 * (1 / zoom), 2 * (1 / zoom), -2 * (1 / zoom), center.X, center.Y, fix.X, fix.Y, zoom, dvec, smoothVal);

	cudaMemcpy(hvec, dvec, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(dvec);

	return cut(hvec, size);
}

#endif