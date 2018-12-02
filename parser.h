#ifndef parser
#define parser

#include <iostream>
#include <cmath>
#include "vector2.h"

using namespace std;

class Args 
{
    public:
        Args();
        ~Args();
        void print(void);
        Vector2 *center;
        Vector2 *fix;
        Vector2 *size;
        bool loop;
        double zoom;
        double zoomEnd;
        double m;
        double stepsize;
        int iterations;
        string folder;
        bool smooth;
        double smoothVal;
};

Args::Args() 
{
    this->center = new Vector2();
    this->fix = new Vector2();
    this->size = new Vector2(1920, 1080);
    this->loop = false;
    this->zoom = 1;
    this->zoomEnd = 10;
    this->m = 1.3;
    this->stepsize = 0.02;
    this->folder = "root";
    this->iterations = 10000;
    this->smooth = false;
    this->smoothVal = log(2);
}

Args::~Args() 
{
    delete this->center;
    delete this->fix;
    delete this->size;
}

void Args::print(void) 
{
    cout << endl << "Arguments: " << endl;

    cout << "   Center: " << (*this->center).X << " | " << (*this->center).Y << endl;
    cout << "   Fix: " << (*this->fix).X << " | " << (*this->fix).Y << endl;
    cout << "   Loop: " << (this->loop ? "true" : "false") << endl;
    cout << "   Zoom: " << this->zoom << endl;
    if (this->loop) cout << "   End Zoom: " << this->zoomEnd << endl;
    cout << "   M: " << this->m << endl;
    cout << "   Step size: " << this->stepsize << endl;
    if (this->folder != "root") cout << "   Folder: " << this->folder << endl;
    cout << "   Iterations: " << this->iterations << endl;
    cout << "   Smooth: " << (this->smooth ? "true" : "false") << endl;
    if (this->smooth) cout << "   Smoothing value: " << this->smoothVal << endl;
}

int find(int argc, char *argv[], string symbol, int exp) 
{
    int index = -1;

    for (int i = 0; i < argc; i++) 
    {
        if (string(argv[i]) == symbol && argc > exp + 1)
        {
            index = i;
            break;
        }
    }
    
    if (index == -1) return -1;

    for (int i = index + 1; i <= index + exp; i++) 
    {
        if (string(argv[0]).length() > 2 && string(argv[i]).at(0) == '-' && string(argv[i]).at(1) == '-') return -1;
    }
    
    return index;
}

Args parse(int argc, char *argv[]) 
{
    Args arg = Args();

    try 
    {
        int index = find(argc, argv, "--center", 2);
        if (index != -1) 
        {
            (*arg.center).X = stod(argv[index + 1]);
            (*arg.center).Y = stod(argv[index + 2]);
        }

        index = find(argc, argv, "--fix", 2);
        if (index != -1) 
        {
            (*arg.fix).X = stod(argv[index + 1]);
            (*arg.fix).Y = stod(argv[index + 2]);
        }

        index = find(argc, argv, "--size", 2);
        if (index != -1) 
        {
            (*arg.size).X = stod(argv[index + 1]);
            (*arg.size).Y = stod(argv[index + 2]);
        }

        index = find(argc, argv, "--zoom", 1);
        if (index != -1) 
        {
            arg.zoom = stod(argv[index + 1]);
        }

        index = find(argc, argv, "--zoomEnd", 1);
        if (index != -1) 
        {
            arg.zoomEnd = stod(argv[index + 1]);
        }

        index = find(argc, argv, "--stepsize", 1);
        if (index != -1) 
        {
            arg.stepsize = stod(argv[index + 1]);
        }

        index = find(argc, argv, "--folder", 1);
        if (index != -1) 
        {
            arg.folder = argv[index + 1];
        }

        index = find(argc, argv, "--iterations", 1);
        if (index != -1) 
        {
            arg.iterations = stoi(argv[index + 1]);
        }

        index = find(argc, argv, "--m", 1);
        if (index != -1) 
        {
            arg.m = stod(argv[index + 1]);
        }

        index = find(argc, argv, "--sv", 1);
        if (index != -1) 
        {
            arg.smoothVal = stod(argv[index + 1]);
        }

        index = find(argc, argv, "--l", 0);
        if (index != -1) arg.loop = true;

        index = find(argc, argv, "--s", 0);
        if (index != -1) arg.smooth = true;

        index = find(argc, argv, "--h", 0);
        if (index != -1) 
        {
            cout << endl << "    --size          height and withe of the image. (double)" << endl << "    --center        x and y of the center of the image. (double)" << endl << "    --fix           x and y of the vector added per zoom. (double)" << endl << "    --zoom          sets the start zoom. (double)" << endl << "    --zoomEnd       sets the end of the zoom. (double)" << endl << "    --l             enables auto loop." << endl << "    --m             the m of the linar funktion used to increase the zoom. (double)" << endl << "    --stepsize      the stepsize used in the loop. (double)" << endl << "    --folder        the folder where to put the images. (string)" << endl << "    --iterations    the iterations per pixel. (int)" << endl << "    --s             enables the smooth color algorithm." << endl << "    --sv            sets the value for the smooth color algorithm. (double)" << endl<< endl;
        }
    }
    catch (...) 
    {
        cout << "Invalid Arguments, use --h for help" << endl;
        exit(2);
    }

    arg.print();
    cin.get();

    return arg;
}

#endif