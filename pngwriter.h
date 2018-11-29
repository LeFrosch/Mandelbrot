#ifndef pngwriter
#define pngwriter

#include <png++/png.hpp>
#include <iostream>
#include "vector2.h"

using namespace std;
using namespace png;

int write(int *vec, int maxIt, Vector2 size, string name, string folder) 
{
    image<rgb_pixel> im((int)size.X, (int)size.Y);
    pixel_buffer<rgb_pixel> pixel(im.get_width(), im.get_height());
    
    int i = 0;
    for (int y = 0; y < im.get_height(); ++y)
    {
        for (int x = 0; x < im.get_width(); ++x)
        {
            i = y * size.X + x;
            if (vec[i] < maxIt)
                pixel.set_pixel(x, y, rgb_pixel(vec[i] % 128 * 2, vec[i] % 32 * 7, vec[i] % 16 * 14));
            else
                pixel.set_pixel(x, y, rgb_pixel(0, 0, 0));
        }
    }

    im.set_pixbuf(pixel);
    if (folder == "root")
        im.write(name);
    else
        im.write(folder + "/" + name);
    
    return 0;
}

int write(int *vec, int maxIt, Vector2 size, int in, int dez, string folder) 
{
    string name = to_string(in);
    for (int i = 0; i < dez - to_string(in).length(); i++)
        name = "0" + name;

    return write(vec, maxIt, size, "image-" + name, folder);
}

#endif