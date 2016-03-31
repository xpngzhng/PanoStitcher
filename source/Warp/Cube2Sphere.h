#pragma once

class CPSCube2Sphere
{
public:
	CPSCube2Sphere(void);
	CPSCube2Sphere(unsigned int height);
	~CPSCube2Sphere(void);

	int Convert2Sphere(const unsigned char* cubeData, int cubeWidth, int cubeHeight, int cubeStep,
		unsigned char* sphereData, int sphereWidth, int sphereHeight, int sphereStep);

	unsigned int CalSphereHeight(const unsigned int cubeHeight);
   
	int GetSphereWidth();
	int GetSphereHeight();


private:
	unsigned char* GetColor(const unsigned char* srcData,double theta,double phi);
	void Initial();

private:
	 int spherewidth, sphereheight;
	 int cubicwidth, cubicheight;
	 double deta;
	 double halfheight;
	 int heightMult4;
	 int heightMult5;
	 int* xpos;
	 int* ypos ;
};
