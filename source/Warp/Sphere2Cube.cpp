#include <iostream>
#include "Sphere2Cube.h"
#include "Resample.h"
#include <cmath>

#ifdef M_PI
#else
#define M_PI 3.1415926
#endif
static	const double D90 = 0.5 * M_PI;
static	const double D270 = 1.5 * M_PI;
static	const double DualPI = 2 * M_PI;

CPSSphere2Cube::CPSSphere2Cube(void)
{
	cubicHeight =0;
	cubicWidth =0;
	spherHeight =0;
	spherWidth =0;
	deat= 0.0;
}

CPSSphere2Cube::~CPSSphere2Cube(void)
{
}

CPSSphere2Cube::CPSSphere2Cube(unsigned int height)
{
	cubicHeight = height;
	cubicWidth = 6 * cubicHeight;
}

int CPSSphere2Cube::GetCubeWidth()
{
	return cubicWidth;
}
int CPSSphere2Cube::GetCubeHeight()
{
	return cubicHeight;
}
//////////////////////////////////////////////////////////////////////////
int CPSSphere2Cube::Convert2Cube(const unsigned char* sphereData, int sphereWidth, int sphereHeight, int sphereStep,
    unsigned char* cubeData, int cudeWidth, int cubeHeight, int cubeStep)
{
	//this->pl = pl;
	register int i = 0;
	register int j = 0;
	int blockb = 0;
	int blocke = 0;
	spherHeight = sphereHeight;
	spherWidth = sphereWidth;
	//根据球形全景高度计算立方体全景高度
	if (cubicHeight<=0)
	{
		cubicHeight=CalCubeHeight(spherHeight);
		cubicWidth=cubicHeight*6;
	}

	deat = spherHeight / M_PI;
	blockb = 0;
	blocke = cubicWidth;
	unsigned char dest[3];

	FPoint pd ;
	pd.x=0.0;
	pd.y=0.0;

	int p=0;
	for ( i = 0; i < cubicHeight; i++)
	{
        for (j = blockb; j < blocke; j++)
        {
            GetPosition((j << 1) + 1, (i << 1) + 1, pd);
            double x = (DualPI - pd.x) * deat - 1;
            double y = pd.y * deat;
            bicubicResample(sphereWidth, sphereHeight, sphereStep, sphereData, x, y, cubeData + i * cubeStep + j * 3);
        }
	}
	return 1;
}

void CPSSphere2Cube::GetPosition(int x,int y,FPoint & pd)
{

	int h = cubicHeight * 2;
	
	int position = x / h;
	x = x % h;
	y = y % h;
	double r = cubicHeight;

	switch (position) 
	{
	   case 0://front
		   warp(r, x-r, y-r,pd);
		   pd.y=pd.y+D90;
		   break;
	   case 1://right
		   warp(r, x-r, y-r,pd);
		   pd.x+=D90;
		   pd.y+=D90;
		   break;
	   case 2://back
		   warp(r, x-r, y-r,pd);
		   pd.x+=M_PI;
		   pd.y+=D90;
		   break;
	   case 3://left
		   warp(r, x-r, y-r,pd);
		   pd.x+=D270;
		   pd.y+=D90;
		   break;
	   case 4://top
		   warp1(r, x-r, y-r,pd);
		   pd.y=D90-pd.y;
		   break;
	   case 5://bottom
		   warp1(r, x-r, r-y,pd);
		   pd.y+=D90;
		   break;
	   default:
		   break;
		  
	}
	pd.x=mod(-pd.x,DualPI);
	pd.y=mod(pd.y,M_PI);

}

/////返回值x 满足  0<=x<c
double CPSSphere2Cube::mod(double v, double c) 
{ 
	if (c != 0) {
		if (v >= 0.0) {
			int n = (int) (v / c);
			return v - n * c;
		} else {
			int n = (int) (v / c) - 1;
			return v - n * c;
		}
	}
	return v;
}

void CPSSphere2Cube::warp(double r, double x, double y,FPoint & res) 
{
	double theta = atan2(x, r);
	double phi = atan(cos(theta)*y/r);
	res.x=theta;
	res.y=phi;
}

void CPSSphere2Cube::warp1(double r, double x, double y,FPoint & res) 
{
	double theta = atan2(x, y);
	double phi = D90 - atan(sqrt(x*x+y*y)/r);
	res.x=theta;
	res.y=phi;
}

unsigned int CPSSphere2Cube::CalCubeHeight(const unsigned int sphereHeight)
{
	return (int) (2 * (float) sphereHeight / M_PI + 0.5);
}

double mod(double v, double c)
{
    if (c != 0) {
        if (v >= 0.0) {
            int n = (int)(v / c);
            return v - n * c;
        }
        else {
            int n = (int)(v / c) - 1;
            return v - n * c;
        }
    }
    return v;
}

void warp(double r, double x, double y, cv::Point2d& res)
{
    double theta = atan2(x, r);
    double phi = atan(cos(theta) * y / r);
    res.x = theta;
    res.y = phi;
}

void warp1(double r, double x, double y, cv::Point2d& res)
{
    double theta = atan2(x, y);
    double phi = D90 - atan(sqrt(x * x + y * y) / r);
    res.x = theta;
    res.y = phi;
}

void cubeToSphereAngle(int x, int y, int cubeHeight, cv::Point2d& pd)
{
    x = x * 2 + 1;
    y = y * 2 + 1;
    int h = cubeHeight * 2;

    int position = x / h;
    x = x % h;
    y = y % h;
    double r = cubeHeight;

    switch (position)
    {
    case 0://front
        warp(r, x - r, y - r, pd);
        pd.y = pd.y + D90;
        break;
    case 1://right
        warp(r, x - r, y - r, pd);
        pd.x += D90;
        pd.y += D90;
        break;
    case 2://back
        warp(r, x - r, y - r, pd);
        pd.x += M_PI;
        pd.y += D90;
        break;
    case 3://left
        warp(r, x - r, y - r, pd);
        pd.x += D270;
        pd.y += D90;
        break;
    case 4://top
        warp1(r, x - r, y - r, pd);
        pd.y = D90 - pd.y;
        break;
    case 5://bottom
        warp1(r, x - r, r - y, pd);
        pd.y += D90;
        break;
    default:
        break;

    }
    pd.x = mod(-pd.x, DualPI);
    pd.y = mod(pd.y, M_PI);

}

void getEquirectToCubeInverseMap(cv::Mat& map, int equirectWidth, int equirectHeight, int cubeWidth, int cubeHeight)
{
    double delta = equirectHeight / M_PI;
    map.create(cubeHeight, cubeWidth, CV_64FC2);

    int p = 0;
    for (int i = 0; i < cubeHeight; i++)
    {
        double* ptr = map.ptr<double>(i);
        for (int j = 0; j < cubeWidth; j++)
        {
            cv::Point2d pd;
            cubeToSphereAngle(j, i, cubeHeight, pd);
            double x = (DualPI - pd.x) * delta - 1;
            double y = pd.y * delta;
            *(ptr++) = x;
            *(ptr++) = y;
        }
    }
}