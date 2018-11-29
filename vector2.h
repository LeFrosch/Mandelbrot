#pragma once

#ifndef vector2
#define vector2

class Vector2
{
	public:
		double X;
		double Y;

		Vector2(double x, double y);
		Vector2();

		void Set(double x, double y);
};

Vector2::Vector2(double x, double y)
{
	this->X = x;
	this->Y = y;
}

Vector2::Vector2() 
{
	this->X = 0;
	this->Y = 0;
}

#endif