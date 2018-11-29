#define ADJUST 1

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "complex.h"
#include "window.h";
#include "parser.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ const double xMin = -2;
__device__ const double xMax = 2;
__device__ const double yMin = -2;
__device__ const double yMax = 2;
__device__ const double zoom = 1;

__global__ void iteration(int maxIt, int length, int width, int height, int *vec)
{
	int loop;
	double c;
	double imZ;
	double rlZ;
	double imC;
	double rlC;
	double xzoom = ((xMax - xMin) / width);
	double yzoom = ((yMax - yMin) / height);

	for (int i = threadIdx.x; i < length; i += blockDim.x + blockIdx.x)
	{
		rlC = xzoom * (i / height) - abs(xMin);
		imC = yzoom * (i % height) - abs(yMin);

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

int main(int argc, char *argv[])
{
	parse(argc, argv);

	cudaFree(0);

	int s;
	cout << "Size: ";
	cin >> s;
	cout << endl;

	Vector2 size(s, s);

	int N = (int)(size.X * size.Y * 4);
	int it = 2000;

	int *hvec = (int*)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++)
		hvec[i] = 200;

	int *dvec = 0;
	cudaMalloc((void**)&dvec, N * sizeof(int));
	cudaMemcpy(dvec, hvec, N * sizeof(int), cudaMemcpyHostToDevice);


	iteration<<<10, 1024>>>(it, N, size.X * 2, size.Y * 2, dvec);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(hvec, dvec, N * sizeof(int), cudaMemcpyDeviceToHost);

	createWindow(size, N, it, hvec);

	cudaFree(hvec);
	cudaFree(dvec);

	return 0;
}