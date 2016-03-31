#include "Cube2Sphere.h"
#include "Resample.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifdef M_PI
#else
#define M_PI 3.1415926
#endif
 static const double dualPI = 2 * M_PI;
 static const double D45 = M_PI / 4.0;
 static const double D90 = M_PI / 2.0;


CPSCube2Sphere::CPSCube2Sphere(void)
{
	Initial();
}

CPSCube2Sphere::CPSCube2Sphere(unsigned int height)
{
    Initial();
	sphereheight = height;
	spherewidth = 2 * height;

	
}

void CPSCube2Sphere::Initial()
{
	spherewidth=0;
	sphereheight=0;
	cubicwidth=0;
	cubicheight=0;
	deta=0.0;
	halfheight=0.0;
	heightMult4=0;
	heightMult5=0;
	xpos=new int[4];
	ypos=new int[4];
}

int CPSCube2Sphere::GetSphereWidth()
{
	return spherewidth;
}

int CPSCube2Sphere::GetSphereHeight()
{
	return sphereheight;
}
/************************************************************************/
/* 
新的立-球转换方法，在数据层操作，无对象依赖
@param cubeData 立方体数据
@param w 立方体宽度
@param h 立方体高度
@sphereData 球数据 该数据在本方法内部申请内存，注意释放
生成球的尺寸通过get方法获取
*/
/************************************************************************/
int CPSCube2Sphere::Convert2Sphere(const unsigned char* cubeData, int cubeWidth, int cubeHeight, int cubeStep,
    unsigned char* sphereData, int sphereWidth, int sphereHeight, int sphereStep)
{
	cubicwidth = cubeWidth;
	cubicheight = cubeHeight;
	if (cubeHeight * 6 != cubeWidth)
	{
		return -1;
	}
	halfheight = cubicheight / 2.0;
	heightMult4 = cubicheight * 4;
	heightMult5 = cubicheight * 5;
	if (sphereheight<=0)
	{
		sphereheight=CalSphereHeight(cubicheight);
		spherewidth=2*sphereheight;
	}
	deta = M_PI / sphereheight;
	double* phi = new double[sphereheight];
	phi[0] = 0.75 * deta;
	for (int i = 1; i < sphereheight; i++) 
	{
		phi[i] = phi[i - 1] + deta;
	}
	double* theta = new double[spherewidth];
	theta[0] = (spherewidth - 0.75) * deta;
	for (int i = 1; i < spherewidth; i++) 
	{
		theta[i] = theta[i - 1] - deta;
	}
	unsigned char* dest=NULL;
	int percent=0;
	for (int i = 0; i < sphereheight; i++) 
	{
		for (int j = 0; j < spherewidth; j++) 
		{
			if((dest=GetColor(cubeData,theta[j], phi[i]))!=NULL)
				memcpy(sphereData+i*spherewidth*3+j*3,dest,3);
		}		
	}
	delete[] theta;
	delete[] phi;
	return 1;
}

unsigned char* CPSCube2Sphere::GetColor(const unsigned char* srcData,double theta,double phi)
{
	theta = dualPI - theta;
	static unsigned char* color=NULL;
	//首先根据球形全景求出立方体全景的六个面
	double xx = 0.0; //double x
	double yy=0.0; //double y
	int region = 0;
	double tanD90subPHI = tan(D90 - phi);
	int side = ((int)floor((theta + D45) / D90)) % 4;
	yy = halfheight * tanD90subPHI / cos(theta - D90 * side);
	if (yy >= halfheight) 
	{ //TOP
		region = 4;
	} 
	else if (yy <= ( -halfheight))
	{ //BOTTOM
		region = 5;
	} 
	else
	{
		region = side;
	}

	double r;
	switch (region) {
		//前后左右四个面，每个面fov为90度
		case 0: //LEFT
		case 1: //FRONT
		case 2: //RIGHT
		case 3: //BACK
			xx = halfheight * (1 + tan(theta - D90 * region));
			yy = halfheight - yy;
			break;
		case 4: //TOP
			r = halfheight / tanD90subPHI;
			xx = r * sin(theta) + halfheight-0.5;
			yy = r * cos(theta) + halfheight-0.5;
			break;
		case 5: //BOTTOM
			r = halfheight / tanD90subPHI;
			xx = halfheight - r * sin(theta);
			yy = halfheight + r * cos(theta);
			break;
	}
	xx = xx < 0 ? region * cubicheight : (xx >= cubicheight ? cubicheight - 1 + region * cubicheight
		: xx + region * cubicheight);
	yy = yy < 0 ? 0 : (yy >= cubicheight ? cubicheight - 1 : yy);
	
	bool done = false;
	int temp = 0;
	int hbp = (int)yy - 1, hep = hbp + 4;
	int wbp = (int)xx - 1, wep = wbp + 4;

	if (hbp >= 0 && hep <= cubicheight) 
	{ 
		if ((wbp >= 0 && wep <= heightMult4) || (wbp >= heightMult4 && wep <= heightMult5) 
			|| (wbp >= heightMult5 && wep <= cubicwidth)) 
		{ 
			temp = 0;
			for (int i = hbp; i < hep; i++)
			{
				ypos[temp++] = i;
			}
			temp = 0;
			for (int i = wbp; i < wep; i++)
			{
				xpos[temp++] = i;
			}
			color  = interp.resampling (cubicwidth,cubicheight,srcData,xpos,ypos);
			done = true;
		}
	}

	if (!done) 
	{
		if (xx >= 0 && xx < heightMult4) 
		{ 
			temp = 0;
			for (int i = hbp; i < hep; i++)
			{
				ypos[temp++] = yy;
			}
			temp = 0;
			for (int i = wbp; i < wep; i++)
			{

				xpos[temp++] = xx;
			}
			color  = interp.resampling (cubicwidth,cubicheight,srcData,xpos,ypos);
		} 
		else if (xx >= heightMult4 && xx < heightMult5) 
		{
			temp = 0;
			for (int i = hbp; i < hep; i++)
			{
				ypos[temp++] = yy;
			}
			temp = 0;
			for (int i = wbp; i < wep; i++)
			{
				xpos[temp++] = xx;
			}
			color  = interp.resampling (cubicwidth,cubicheight,srcData,xpos,ypos);
		} 
		else if (xx >= heightMult5 && xx < cubicwidth) 
		{ 
			temp = 0;
			for (int i = hbp; i < hep; i++)
			{
				ypos[temp++] = yy;
			}
			temp = 0;
			for (int i = wbp; i < wep; i++)
			{
				xpos[temp++] = xx;
			}
			color  = interp.resampling (cubicwidth,cubicheight,srcData,xpos,ypos);
		} 
	}
	return color;
}

unsigned int CPSCube2Sphere::CalSphereHeight(const unsigned int cubeHeight)
{
   return (int) (M_PI * (float) cubeHeight/ 2 + 0.5);
}


CPSCube2Sphere::~CPSCube2Sphere(void)
{
	delete[] xpos;
	delete[] ypos;
}
