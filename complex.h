#pragma once
#ifndef complex
#define complex

__device__ __host__ void Norm(double *c, double imaginary, double real)
{
	*c = imaginary * imaginary + real * real;
}

__device__ __host__ void Pow(double *imaginary, double *real)
{
	double temp = *real;

	*real = *real * *real - *imaginary * *imaginary;
	*imaginary = 2 * temp * *imaginary;
}

__device__ __host__ void Add(double *imaginaryZ, double *realZ, double imaginaryC, double realC)
{
	*realZ = *realZ + realC;
	*imaginaryZ = *imaginaryZ + imaginaryC;
}

#endif