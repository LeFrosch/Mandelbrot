#include <iostream>
#include "kernel.h"
#include "window.h"
#include "pngwriter.h"
#include "parser.h"

using namespace std;

const int iterations = 1000;

int main(int argc, char *argv[])
{
    Vector2 size(4096, 2160);
    // ./Fractal --center 500 0 --fix -189.02448 -37.13268 --zoom 200000

    Args arg = parse(argc, argv);

    if (!arg.loop) 
    {
        int *vec = mandelbrot(iterations, arg.zoom, size, *arg.center, *arg.fix);
        write(vec, iterations, size, "mandelbrot", arg.folder);
    }
    else 
    {
        int count = 0;
        for(double zoom = arg.zoom; zoom < arg.zoomEnd; zoom += arg.stepsize * arg.m * zoom)  
        {
            cout << zoom << " out of " << arg.zoomEnd << endl;
            int *vec = mandelbrot(iterations, zoom, size, *arg.center, *arg.fix);
            write(vec, iterations, size, count++, 6, arg.folder);
            free(vec);
        }
    }

    return 0;
}