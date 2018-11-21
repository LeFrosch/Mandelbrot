#pragma once
#ifndef window
#define window

#include <windows.h>
#include <windowsx.h>
#include "vector2.h"
#include <SFML\Graphics.hpp>

using namespace std;

int createWindow(Vector2 size, int length, int itartations, int *vec)
{
	sf::RenderWindow win(sf::VideoMode(size.X, size.Y), "Fractal");

	sf::Texture texture;
	texture.create(size.X * 2, size.Y * 2);

	sf::Sprite sprite(texture);

	sprite.setScale(0.5, 0.5);

	sf::Uint8 *pixels = new sf::Uint8[length * 4];

	for (int i = 0; i < length * 4; i += 4)
	{
		if (vec[i / 4] < itartations)
		{
			pixels[i] = vec[i / 4] % 128 * 2;
			pixels[i + 1] = vec[i / 4] % 32 * 7;
			pixels[i + 2] = vec[i / 4] % 16 * 14;
			pixels[i + 3] = 255;
		}
		else
		{
			pixels[i] = 0;
			pixels[i + 1] = 0;
			pixels[i + 2] = 0;
			pixels[i + 3] = 255;
		}
	}

	while (win.isOpen())
	{
		sf::Event event;

		while (win.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				win.close();
		}

		texture.update(pixels);
		win.draw(sprite);
		win.display();
	}

	delete[] pixels;

	return 0;
}

#endif