#pragma once
#ifndef gussianfilter
#define gussianfilter

#include <png++/png.hpp>
#include <iostream>
#include <cmath>
#include "vector2.h"

using namespace png;
using namespace std;

const double matrix[25] 
{
    0, 1, 2, 1, 0,
    1, 3, 5, 3, 1,
    2, 5, 9, 5, 2,
    1, 3, 5, 3, 1,
    0, 1, 2, 1, 0
};

double* getMatrix() 
{
    double *guassian = (double*)malloc(25 * sizeof(double));

    for (int i = 0; i < 25; i++) 
    {
        guassian[i] = matrix[i] / 57; 
    }

    return guassian;
}

__global__ void blure(double matrix[], double result[], double input[], int width, int height)
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

pixel_buffer<rgb_pixel> kernalcall(pixel_buffer<rgb_pixel> *in) 
{
    double *matrix = getMatrix();
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

    blure<<<16, 1024>>>(dmatrix, doutput, dinput, width, height);

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

void openImage(string path, string folder) 
{
    image<rgb_pixel> im(path);
    pixel_buffer<rgb_pixel> buffer = im.get_pixbuf();
    
    int x = 2000;
    int y = 50;

    rgb_pixel p = buffer.get_pixel(x, y);

    pixel_buffer<rgb_pixel> result = kernalcall(&buffer);

    im.set_pixbuf(result);

    if (folder == "root")
        im.write(path);
    else
        im.write(folder + "/" + path);
}

#endif