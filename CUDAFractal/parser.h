#pragma once
#ifndef parser
#define parser

#include <iostream>
#include "vector2.h"

using namespace std;

class Args
{
	public:
		Args();

		Vector2 size;
		Vector2 center;
		Vector2 fix;
		Vector2 zoom;
};

Args::Args() 
{
	this->size = Vector2(1000, 1000);
	this->center = Vector2(0, 0);
	this->fix = Vector2(500, 500);
	this->zoom = Vector2(1, 1);
}

int findInArgs(int length, char *argv[], char *search)
{
	for (int i = 0; i < length; i++)
	{
		if (argv[i] == *search || argv[i] == "-" + *search || argv[i] == "/" + *search)
			return i;
	}

	return -1;
}

Args parse(int argc, char *argv[])
{
	Args arg = Args();

	cout << findInArgs(argc, argv, "lola");

	return arg;
}

#endif