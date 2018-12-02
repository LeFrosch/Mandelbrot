#include <iostream>
#include "kernel.h"
#include "pngwriter.h"
#include "parser.h"

using namespace std;

inline bool exists (const string name) {
    ifstream f(name.c_str());
    return f.good();
}

// ./Fractal --size 4096 2160 --center 500 0 --fix -55.07529 167.82905 --zoomEnd 4096000 --iterations 15000 --s --sv 250 --rgb 1.0 0.05 2.0 --l --stepsize 0.0025 --folder images 


int main(int argc, char *argv[])
{
    Args arg = parse(argc, argv);
    clock_t start;
    clock_t end;

    if (!arg.loop) 
    {
        double *vec = mandelbrot(arg.iterations, arg.zoom, *arg.size, *arg.center, *arg.fix, arg.smooth, arg.smoothVal);
        write(vec, arg.iterations, *arg.size, "mandelbrot", arg.folder, arg.smooth, arg.r, arg.g, arg.b);
    }
    else 
    {
        int count = 0;
        for(double zoom = arg.zoom; zoom < arg.zoomEnd; zoom += arg.stepsize * arg.m * zoom)  
        {
            if (arg.skip && exists("image-" + getName(count, 6) + ".png")) 
            {
                cout << zoom << " out of " << arg.zoomEnd << " ";
                start = clock();
                double *vec = mandelbrot(arg.iterations, zoom, *arg.size, *arg.center, *arg.fix, arg.smooth, arg.smoothVal);
                write(vec, arg.iterations, *arg.size, count, 6, arg.folder, arg.smooth, arg.r, arg.g, arg.b);
                free(vec);
                end = clock();
                cout << "time: " << ((double)(end-start) / CLOCKS_PER_SEC) << "ms";
            }
            else 
            {
                cout << "skipped image" << endl;
            }
            count++;
        }
    }

    return 0;
}
