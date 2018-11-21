#define ADJUST 1

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "complex.h"
#include "window.h";

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

__global__ void iteration(int maxIt, int length, int width, int height, int *vec, bool *abort)
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
		if (*abort) break;

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

int main()
{
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
	
	bool *habort = new bool;
	bool *dabort = new bool;
	cudaHostAlloc(&habort, sizeof(bool), cudaHostAllocDefault);
	cudaMalloc(&dabort, sizeof(bool));

	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	cudaMemcpy(dabort, habort, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dvec, hvec, N * sizeof(int), cudaMemcpyHostToDevice);

	int *test = new int[1];
	memset(test, 0, sizeof(int));

	test[0] = 1;

	cudaMemcpyToSymbol(cabort, test, sizeof(int));

	iteration<<<10, 1024, 0, stream>>>(it, N, size.X * 2, size.Y * 2, dvec, dabort);
	//Sleep(100);


	test[0] = 0;
	cudaMemcpyToSymbolAsync(cabort, test, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	//cudaMemsetAsync(habort, true, sizeof(bool), stream);
	//cudaMemcpyAsync(dabort, habort, sizeof(bool), cudaMemcpyHostToDevice, stream);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(hvec, dvec, N * sizeof(int), cudaMemcpyDeviceToHost);

	createWindow(size, N, it, hvec);

	cudaStreamDestroy(stream);
	cudaFree(hvec);
	cudaFree(dvec);
	cudaFree(habort);
	cudaFree(dabort);

	return 0;
}