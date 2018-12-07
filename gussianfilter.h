#pragma once
#ifndef gussianfilter
#define gussianfilter

#include <png++/png.hpp>
#include <iostream>
#include <cmath>
#include "vector2.h"

using namespace png;
using namespace std;

const double matrix1[25] 
{
    0, 1, 2, 1, 0,
    1, 3, 5, 3, 1,
    2, 5, 9, 5, 2,
    1, 3, 5, 3, 1,
    0, 1, 2, 1, 0
};

const double matrix0[25] 
{
    0.0, 0.5, 1.0, 0.5, 0.0,
    0.5, 1.5, 2.5, 1.5, 0.5,
    1.0, 2.5, 4.5, 2.5, 1.0,
    0.5, 1.5, 2.5, 1.5, 0.5,
    0.0, 0.5, 1.0, 0.5, 0.0
};

#pragma region 2D

double* get2DMatrix()
{
    double *guassian = (double*)malloc(25 * sizeof(double));
    double sum = 0;

    for (int i = 0; i < 25; i++) 
    {
        sum += matrix1[i];
    }

    for (int i = 0; i < 25; i++) 
    {
        guassian[i] = matrix1[i] / sum; 
    }

    return guassian;
}

__global__ void blure2D(double matrix[], double result[], double input[], int width, int height)
{
    int stride = blockDim.x * gridDim.x;
    int max = width * height * 3;

    for (int i = (threadIdx.x + blockIdx.x * blockDim.x) * 3; i < max; i += stride * 3) 
    {
        int x = ((double)i / 3) / (double)height;
        int y = (int)((double)i / 3) % height;
        if (x < 2 || x > width - 2 || y < 2 || y > height - 2) continue;

        double rsum = 0;
        double gsum = 0;
        double bsum = 0;

        for (int px = -2; px <= 2; px++) 
        {
            for (int py = -2; py <= 2; py++) 
            {
                double g = matrix[5 * (px + 2) + (py + 2)];

                rsum += input[height * 3 * (x + px) + (y + py) * 3] * g;
                gsum += input[height * 3 * (x + px) + (y + py) * 3 + 1] * g;
                bsum += input[height * 3 * (x + px) + (y + py) * 3 + 2] * g;
            }
        }

        result[i] = rsum;
        result[i + 1] = gsum;
        result[i + 2] = bsum;
    }
}

pixel_buffer<rgb_pixel> kernalcall2D(pixel_buffer<rgb_pixel> *in) 
{
    double *matrix = get2DMatrix();
    int width = (*in).get_width();
    int height = (*in).get_height();

    int n = width * height * 3;
    size_t size = n * sizeof(double);

    double *hinput = (double*)malloc(size);
    double *houtput = (double*)malloc(size);
    double *dinput = 0;
    cudaMalloc(&dinput, size);
    double *doutput = 0;
    cudaMalloc(&doutput, size);
    double *dmatrix = 0;
    cudaMalloc(&dmatrix, 25 * sizeof(double));

    for (int i = 0; i < n; i += 3) 
    {
        int x = (double)(i / 3) / (double)height;
        int y = (i / 3) % height;

        rgb_pixel p = (*in).get_pixel(x, y);
        hinput[i] = p.red;
        hinput[i + 1] = p.green;
        hinput[i + 2] = p.blue;
    }

    cudaMemcpy(dinput, hinput, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmatrix, matrix, sizeof(double) * 25, cudaMemcpyHostToDevice);

    blure2D<<<16, 1024>>>(dmatrix, doutput, dinput, width, height);

    cudaMemcpy(houtput, doutput, size, cudaMemcpyDeviceToHost);

    free(hinput);
    free(matrix);
    cudaFree(dinput);
    cudaFree(dmatrix);

    pixel_buffer<rgb_pixel> result = pixel_buffer<rgb_pixel>(width, height);

    for (int i = 0; i < n; i += 3) 
    {
        int x = (double)(i / 3) / (double)height;
        int y = (i / 3) % height;

        result.set_pixel(x, y, rgb_pixel(houtput[i], houtput[i + 1], houtput[i + 2]));
    }

    free(houtput);
    cudaFree(doutput);

    return result;
}

#pragma endregion

#pragma region 3D

double* get3DMatrix()
{
    double *guassian = (double*)malloc(50 * sizeof(double));
    double sum = 0;

    for (int i = 0; i < 25; i++) 
    {
        sum += matrix0[i] + matrix1[i];
    }

    sum /= 4;

    for (int i = 0; i < 25; i++) 
    {
        guassian[i] = matrix0[i] / sum;
        guassian[i + 25] = matrix1[i] / sum; 
    }

    return guassian;
}

__global__ void blure3D(double matrix[], double result[], double input0[], double input1[], int width, int height)
{
    int stride = blockDim.x * gridDim.x;
    int max = width * height * 3;

    for (int i = (threadIdx.x + blockIdx.x * blockDim.x) * 3; i < max; i += stride * 3) 
    {
        int x = ((double)i / 3) / (double)height;
        int y = (int)((double)i / 3) % height;
        if (x < 2 || x > width - 2 || y < 2 || y > height - 2) continue;

        double rsum = 0;
        double gsum = 0;
        double bsum = 0;

        for (int px = -2; px <= 2; px++) 
        {
            for (int py = -2; py <= 2; py++) 
            {
                double g0 = matrix[5 * (px + 2) + (py + 2)];
                double g1 = matrix[5 * (px + 2) + (py + 2) + 25];
                int index = height * 3 * (x + px) + (y + py) * 3;

                rsum += input0[index] * g0;
                gsum += input0[index + 1] * g0;
                bsum += input0[index + 2] * g0;

                rsum += input1[index] * g1;
                gsum += input1[index + 1] * g1;
                bsum += input1[index + 2] * g1;
            }
        }

        result[i] = rsum;
        result[i + 1] = gsum;
        result[i + 2] = bsum;
    }
}

pixel_buffer<rgb_pixel> kernalcall3D(pixel_buffer<rgb_pixel> *in0, pixel_buffer<rgb_pixel> *in1) 
{
    double *matrix = get3DMatrix();
    int width = (*in1).get_width();
    int height = (*in1).get_height();

    int n = width * height * 3;
    size_t size = n * sizeof(double);

    double *hinput0 = (double*)malloc(size);
    double *hinput1 = (double*)malloc(size);
    double *houtput = (double*)malloc(size);
    double *dinput0 = 0;
    cudaMalloc(&dinput0, size);
    double *dinput1 = 0;
    cudaMalloc(&dinput1, size);
    double *doutput = 0;
    cudaMalloc(&doutput, size);
    double *dmatrix = 0;
    cudaMalloc(&dmatrix, 50 * sizeof(double));

    for (int i = 0; i < n; i += 3) 
    {
        int x = (double)(i / 3) / (double)height;
        int y = (i / 3) % height;

        rgb_pixel p = (*in0).get_pixel(x, y);
        hinput0[i] = p.red;
        hinput0[i + 1] = p.green;
        hinput0[i + 2] = p.blue;

        p = (*in1).get_pixel(x, y);
        hinput1[i] = p.red;
        hinput1[i + 1] = p.green;
        hinput1[i + 2] = p.blue;
    }

    cudaMemcpy(dinput0, hinput0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dinput1, hinput1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmatrix, matrix, sizeof(double) * 25, cudaMemcpyHostToDevice);

    blure3D<<<16, 1024>>>(dmatrix, doutput, dinput0, dinput1, width, height);

    cudaMemcpy(houtput, doutput, size, cudaMemcpyDeviceToHost);

    free(hinput0);
    free(hinput1);
    free(matrix);
    cudaFree(dinput0);
    cudaFree(dinput1);
    cudaFree(dmatrix);

    pixel_buffer<rgb_pixel> result = pixel_buffer<rgb_pixel>(width, height);

    for (int i = 0; i < n; i += 3) 
    {
        int x = (double)(i / 3) / (double)height;
        int y = (i / 3) % height;

        result.set_pixel(x, y, rgb_pixel(houtput[i], houtput[i + 1], houtput[i + 2]));
    }

    free(houtput);
    cudaFree(doutput);

    return result;
}

#pragma endregion

void openImage(string path, string folder) 
{
    image<rgb_pixel> im(path);
    pixel_buffer<rgb_pixel> buffer = im.get_pixbuf();

    pixel_buffer<rgb_pixel> result = kernalcall2D(&buffer);

    im.set_pixbuf(result);

    if (folder == "root")
        im.write(path);
    else
    {
        size_t index = path.find_last_of("/\\");
        im.write(folder + "/" + path.substr(index + 1));
    }
}

void open2Images(string path0, string path1, string folder) 
{
    // lazy solution, feel free to fix this
    image<rgb_pixel> im0(path0);
    image<rgb_pixel> im1(path1);
    pixel_buffer<rgb_pixel> buffer0 = im0.get_pixbuf();
    pixel_buffer<rgb_pixel> buffer1 = im1.get_pixbuf();

    pixel_buffer<rgb_pixel> result = kernalcall3D(&buffer0, &buffer1);

    im1.set_pixbuf(result);

    if (folder == "root")
        im1.write(path1);
    else
    {
        size_t index = path1.find_last_of("/\\");
        im1.write(folder + "/" + path1.substr(index + 1));
    }
}

#endif