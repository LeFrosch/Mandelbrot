#ifndef parser
#define parser

#include <iostream>
#include "vector2.h"

using namespace std;

class Args 
{
    public:
        Args();
        ~Args();
        Vector2 *center;
        Vector2 *fix;
        bool loop;
        double zoom;
        double zoomEnd;
        double m;
        double stepsize;
        string folder;
};

Args::Args() 
{
    this->center = new Vector2();
    this->fix = new Vector2();
    this->loop = false;
    this->zoom = 1;
    this->zoomEnd = 10;
    this->m = 1.3;
    this->stepsize = 0.02;
    this->folder = "root";
}

Args::~Args() 
{
    delete this->center;
    delete this->fix;
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

        index = find(argc, argv, "--m", 1);
        if (index != -1) 
        {
            arg.m = stod(argv[index + 1]);
        }

        index = find(argc, argv, "--l", 0);
        if (index != -1) arg.loop = true;

        index = find(argc, argv, "--h", 0);
        if (index != -1) 
        {
            cout << endl << "    --center        x and y of the center of the image. (double)" << endl << "    --fix           x and y of the vector added per zoom. (double)" << endl << "    --zoom          sets the start zoom. (double)" << endl << "    --zoomEnd       sets the end of the zoom. (double)" << endl << "    --l             enables auto loop." << endl << "    --m             the m of the linar funktion used to increase the zoom." << endl << "    --stepsize      the stepsize used in the loop." << endl << "    --folder        the folder where to put the images." << endl << endl;
        }
    }
    catch (...) 
    {
        cout << "Invalid Arguments, use --h for help" << endl;
        exit(2);
    }

    return arg;
}

#endif