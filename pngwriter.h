#ifndef pngwriter
#define pngwriter

#include <png++/png.hpp>
#include <iostream>
#include <cmath>
#include "vector2.h"

using namespace std;
using namespace png;

const double a = 3.321928095;
const double b = 0.782985961;
const double c = 0.413668099;

void normal(pixel_buffer<rgb_pixel> *pix, double *vec, int maxIt, Vector2 size) 
{
    int i = 0;
    for (int y = 0; y < (*pix).get_height(); ++y)
    {
        for (int x = 0; x < (*pix).get_width(); ++x)
        {
            i = y * size.X + x;
            if (vec[i] < maxIt)
                (*pix).set_pixel(x, y, rgb_pixel((int)vec[i] % 128 * 2, (int)vec[i] % 32 * 7, (int)vec[i] % 16 * 14));
            else
                (*pix).set_pixel(x, y, rgb_pixel(0, 0, 0));
        }
    }
}

void smoo(pixel_buffer<rgb_pixel> *pix, double *vec, int maxIt, Vector2 size) 
{
    int i;
    for (int y = 0; y < (*pix).get_height(); ++y)
    {
        for (int x = 0; x < (*pix).get_width(); ++x)
        {
            i = y * size.X + x;

            (*pix).set_pixel(x, y, rgb_pixel(255 * (1 - cos(a * vec[i])), 255 * (1 - cos(b * vec[i])), 255 * (1 - cos(c * vec[i]))));
        }
    }
}


int write(double *vec, int maxIt, Vector2 size, string name, string folder, bool smooth) 
{
    image<rgb_pixel> im((int)size.X, (int)size.Y);
    pixel_buffer<rgb_pixel> pixel(im.get_width(), im.get_height());
    
    if (smooth)
        smoo(&pixel, vec, maxIt, size);
    else
        normal(&pixel, vec, maxIt, size);

    im.set_pixbuf(pixel);
    if (folder == "root")
        im.write(name);
    else
        im.write(folder + "/" + name);
    
    return 0;
}

int write(double *vec, int maxIt, Vector2 size, int in, int dez, string folder, bool smooth) 
{
    string name = to_string(in);
    for (int i = 0; i < dez - to_string(in).length(); i++)
        name = "0" + name;

    return write(vec, maxIt, size, "image-" + name, folder, smooth);
}

#endif