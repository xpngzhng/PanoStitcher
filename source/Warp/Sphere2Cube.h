#pragma once

#include "opencv2/core/core.hpp"

class CPSSphere2Cube
{
public:
	typedef struct  
	{
		double x;
		double y;
	}FPoint;

public:
	CPSSphere2Cube(void);
	
	CPSSphere2Cube(unsigned int height);

	int Convert2Cube (const unsigned char* sphereData, int sphereWidth, int sphereHeight, int sphereStep,
		unsigned char* cubeData, int cudeWidth, int cubeHeight, int cubeStep);

	unsigned int CalCubeHeight(const unsigned int sphereHeight);

	int GetCubeWidth();
	int GetCubeHeight();

public:
	~CPSSphere2Cube(void);
private:
	void GetPosition(int x,int y,FPoint & pd);
	double mod(double v, double c);
	void warp(double r, double x, double y,FPoint & res);
	void warp1(double r, double x, double y,FPoint & res);
	
private:
	unsigned int cubicHeight;
	unsigned int cubicWidth;
	unsigned int spherHeight;
	unsigned int spherWidth;
	double deat;
};

void getEquirectToCubeInverseMap(cv::Mat& map, int equirectWidth, int equirectHeight, int cubeWidth, int cubeHeight);
