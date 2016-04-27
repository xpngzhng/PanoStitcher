#include "ZReproject.h"

static const double PI = 3.14159265358979323846264338327950288;
#define DEG_TO_RAD(x) ((x) * 2.0 * PI / 360.0)

static const double R_EPS = 1.0e-6;
static const int MAXITER = 100;

static void matrix_matrix_mult(double m1[3][3], double m2[3][3], double result[3][3])
{
    int i, k;

    for (i = 0; i < 3; i++)
    {
        for (k = 0; k < 3; k++)
        {
            result[i][k] = m1[i][0] * m2[0][k] + m1[i][1] * m2[1][k] + m1[i][2] * m2[2][k];
        }
    }
}

static void SetMatrix(double a, double b, double c, double m[3][3], int cl)
{
    double mx[3][3], my[3][3], mz[3][3], dummy[3][3];
    // Calculate Matrices;
    mx[0][0] = 1.0; 				mx[0][1] = 0.0; 				mx[0][2] = 0.0;
    mx[1][0] = 0.0; 				mx[1][1] = cos(a); 			    mx[1][2] = sin(a);
    mx[2][0] = 0.0;				    mx[2][1] = -mx[1][2];			mx[2][2] = mx[1][1];

    my[0][0] = cos(b); 				my[0][1] = 0.0; 				my[0][2] = -sin(b);
    my[1][0] = 0.0; 				my[1][1] = 1.0; 				my[1][2] = 0.0;
    my[2][0] = -my[0][2];			my[2][1] = 0.0;				    my[2][2] = my[0][0];

    mz[0][0] = cos(c); 			    mz[0][1] = sin(c); 			    mz[0][2] = 0.0;
    mz[1][0] = -mz[0][1]; 			mz[1][1] = mz[0][0]; 			mz[1][2] = 0.0;
    mz[2][0] = 0.0;				    mz[2][1] = 0.0;				    mz[2][2] = 1.0;

    if (cl)
        matrix_matrix_mult(mz, mx, dummy);
    else
        matrix_matrix_mult(mx, mz, dummy);
    matrix_matrix_mult(dummy, my, m);
}

static void squareZero(double *a, int *n, double *root)
{
    if (a[2] == 0.0)
    { // linear equation
        if (a[1] == 0.0)
        { // constant
            if (a[0] == 0.0)
            {
                *n = 1; root[0] = 0.0;
            }
            else
            {
                *n = 0;
            }
        }
        else
        {
            *n = 1; root[0] = -a[0] / a[1];
        }
    }
    else
    {
        if (4.0 * a[2] * a[0] > a[1] * a[1])
        {
            *n = 0;
        }
        else
        {
            *n = 2;
            root[0] = (-a[1] + sqrt(a[1] * a[1] - 4.0 * a[2] * a[0])) / (2.0 * a[2]);
            root[1] = (-a[1] - sqrt(a[1] * a[1] - 4.0 * a[2] * a[0])) / (2.0 * a[2]);
        }
    }

}

static double cubeRoot(double x)
{
    if (x == 0.0)
        return 0.0;
    else if (x > 0.0)
        return pow(x, 1.0 / 3.0);
    else
        return -pow(-x, 1.0 / 3.0);
}

static void cubeZero(double *a, int *n, double *root)
{
    if (a[3] == 0.0)
    { // second order polynomial
        squareZero(a, n, root);
    }
    else
    {
        double p = ((-1.0 / 3.0) * (a[2] / a[3]) * (a[2] / a[3]) + a[1] / a[3]) / 3.0;
        double q = ((2.0 / 27.0) * (a[2] / a[3]) * (a[2] / a[3]) * (a[2] / a[3]) - (1.0 / 3.0) * (a[2] / a[3]) * (a[1] / a[3]) + a[0] / a[3]) / 2.0;

        if (q*q + p*p*p >= 0.0)
        {
            *n = 1;
            root[0] = cubeRoot(-q + sqrt(q*q + p*p*p)) + cubeRoot(-q - sqrt(q*q + p*p*p)) - a[2] / (3.0 * a[3]);
        }
        else
        {
            double phi = acos(-q / sqrt(-p*p*p));
            *n = 3;
            root[0] = 2.0 * sqrt(-p) * cos(phi / 3.0) - a[2] / (3.0 * a[3]);
            root[1] = -2.0 * sqrt(-p) * cos(phi / 3.0 + PI / 3.0) - a[2] / (3.0 * a[3]);
            root[2] = -2.0 * sqrt(-p) * cos(phi / 3.0 - PI / 3.0) - a[2] / (3.0 * a[3]);
        }
    }
    // PrintError("%lg, %lg, %lg, %lg root = %lg", a[3], a[2], a[1], a[0], root[0]);
}

static double smallestRoot(double *p)
{
    int n, i;
    double root[3], sroot = 1000.0;

    cubeZero(p, &n, root);

    for (i = 0; i<n; i++)
    {
        // PrintError("Root %d = %lg", i,root[i]);
        if (root[i] > 0.0 && root[i] < sroot)
            sroot = root[i];
    }

    // PrintError("Smallest Root  = %lg", sroot);
    return sroot;
}

void Remap::clear()
{
    memset(this, 0, sizeof(*this));
}

void Remap::init(const PhotoParam& param_, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    this->srcTX = srcWidth / 2;
    this->srcTY = srcHeight / 2;

    bool fullImage = (param_.imageType == PhotoParam::ImageTypeRectlinear) || (param_.imageType == PhotoParam::ImageTypeFullFrameFishEye);
    PhotoParam param = param_;
    if (fullImage)
    {
        param.cropX = 0;
        param.cropY = 0;
        param.cropWidth = dstWidth;
        param.cropHeight = dstHeight;
    }

    if (param.imageType == PhotoParam::ImageTypeRectlinear)
        this->srcImageType = PTImageTypeRectlinear;
    else if (param.imageType == PhotoParam::ImageTypeFullFrameFishEye)
        this->srcImageType = PTImageTypeFullFrameFishEye;
    else if (param.imageType == PhotoParam::ImageTypeDrumFishEye ||
        param.imageType == PhotoParam::ImageTypeCircularFishEye)
        this->srcImageType = PTImageTypeCircularFishEye;

    this->destTX = param.cropX + param.cropWidth / 2;
    this->destTY = param.cropY + param.cropHeight / 2;

    if (param.shiftX != 0)
    {
        this->mp.horizontal = param.shiftX;
    }
    else
    {
        this->mp.horizontal = 0;
    }
    if (param.shiftY != 0)
    {
        this->mp.vertical = param.shiftY;
    }
    else
    {
        this->mp.vertical = 0;
    }

    double a = DEG_TO_RAD(param.hfov);
    double b = DEG_TO_RAD(360);

    SetMatrix(-DEG_TO_RAD(param.pitch), 0.0, -DEG_TO_RAD(param.roll), this->mp.mt, 0);

    this->mp.distance = ((double)srcWidth) / b;

    if (srcImageType == PTImageTypeRectlinear)
        this->mp.scale[0] = ((double)param.cropWidth) / (2.0 * tan(a / 2.0)) / mp.distance;
    else
        this->mp.scale[0] = ((double)param.cropWidth) / a / this->mp.distance;

    this->mp.scale[1] = this->mp.scale[0];

    this->mp.shear[0] = param.shearX / param.cropHeight;
    this->mp.shear[1] = param.shearY / param.cropWidth;

    this->mp.rot[0] = this->mp.distance*PI;
    this->mp.rot[1] = -param.yaw * this->mp.distance * PI / 180.0;

    this->mp.perspect[0] = (void*)(this->mp.mt);
    this->mp.perspect[1] = (void*)&(this->mp.distance);

    double radial_params[3][5];
    for (int i = 0; i < 3; i++)
    {
        radial_params[i][0] = 1.0;
        radial_params[i][4] = 1000.0;
        for (int k = 1; k < 4; k++)
        {
            radial_params[i][k] = 0.0;
        }
    }

    if (param.beta != 0.0 || param.gamma != 0.0 || param.alpha != 0.0)
    {
        radial_params[0][3] = radial_params[1][3] = radial_params[2][3] = param.alpha;
        radial_params[0][2] = radial_params[1][2] = radial_params[2][2] = param.beta;
        radial_params[0][1] = radial_params[1][1] = radial_params[2][1] = param.gamma;
        double d = 1.0 - (param.alpha + param.beta + param.gamma);
        radial_params[0][0] = radial_params[1][0] = radial_params[2][0] = d;
    }

    double temp[4];
    for (int i = 0; i < 3; i++)
    {
        for (int k = 0; k < 4; k++)
        {
            temp[k] = 0.0;//1.0e-10;
            if (radial_params[i][k] != 0.0)
            {
                temp[k] = (k + 1) * radial_params[i][k];
            }
        }
        radial_params[i][4] = smallestRoot(temp);
    }

    for (int i = 0; i < 4; i++)
        this->mp.rad[i] = radial_params[0][i];

    this->mp.rad[5] = radial_params[0][4];
    this->mp.rad[4] = ((double)(param.cropWidth < param.cropHeight ? param.cropWidth : param.cropHeight)) / 2.0;
}

void Remap::initInverse(const PhotoParam& param_, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    this->destTX = srcWidth / 2;
    this->destTY = srcHeight / 2;

    bool fullImage = (param_.imageType == 0) || (param_.imageType == 1);
    PhotoParam param = param_;
    if (fullImage)
    {
        param.cropX = 0;
        param.cropY = 0;
        param.cropWidth = dstWidth;
        param.cropHeight = dstHeight;
    }

    if (param.imageType == PhotoParam::ImageTypeRectlinear)
        this->srcImageType = PTImageTypeRectlinear;
    else if (param.imageType == PhotoParam::ImageTypeFullFrameFishEye)
        this->srcImageType = PTImageTypeFullFrameFishEye;
    else if (param.imageType == PhotoParam::ImageTypeDrumFishEye ||
        param.imageType == PhotoParam::ImageTypeCircularFishEye)
        this->srcImageType = PTImageTypeCircularFishEye;

    this->srcTX = param.cropX + param.cropWidth / 2;
    this->srcTY = param.cropY + param.cropHeight / 2;

    if (param.shiftX != 0)
    {
        this->mp.horizontal = -param.shiftX;
    }
    else
    {
        this->mp.horizontal = 0;
    }
    if (param.shiftY != 0)
    {
        this->mp.vertical = -param.shiftY;
    }
    else
    {
        this->mp.vertical = 0;
    }

    double a = DEG_TO_RAD(param.hfov);
    double b = DEG_TO_RAD(360);

    SetMatrix(DEG_TO_RAD(param.pitch), 0.0, DEG_TO_RAD(param.roll), this->mp.mt, 1);//为什么yam不用来计算选择矩阵
    this->mp.distance = ((double)srcWidth) / b;
    this->mp.scale[0] = ((double)param.cropWidth) / a / this->mp.distance;
    this->mp.scale[1] = this->mp.scale[0];

    this->mp.shear[0] = -param.shearX / param.cropHeight;
    this->mp.shear[1] = -param.shearY / param.cropWidth;

    this->mp.scale[0] = 1.0 / this->mp.scale[0];
    this->mp.scale[1] = this->mp.scale[0];

    this->mp.rot[0] = this->mp.distance*PI;
    this->mp.rot[1] = param.yaw * this->mp.distance * PI / 180.0;

    this->mp.perspect[0] = (void*)(this->mp.mt);
    this->mp.perspect[1] = (void*)&(this->mp.distance);


    double radial_params[3][5];
    for (int i = 0; i < 3; i++)
    {
        radial_params[i][0] = 1.0;
        radial_params[i][4] = 1000.0;
        for (int k = 1; k < 4; k++)
        {
            radial_params[i][k] = 0.0;
        }
    }

    if (param.beta != 0.0 || param.gamma != 0.0 || param.alpha != 0.0)
    {
        radial_params[0][3] = radial_params[1][3] = radial_params[2][3] = param.alpha;
        radial_params[0][2] = radial_params[1][2] = radial_params[2][2] = param.beta;
        radial_params[0][1] = radial_params[1][1] = radial_params[2][1] = param.gamma;
        double d = 1.0 - (param.alpha + param.beta + param.gamma);
        radial_params[0][0] = radial_params[1][0] = radial_params[2][0] = d;
    }

    double temp[4];
    for (int i = 0; i < 3; i++)
    {
        for (int k = 0; k < 4; k++)
        {
            temp[k] = 0.0;//1.0e-10;
            if (radial_params[i][k] != 0.0)
            {
                temp[k] = (k + 1) * radial_params[i][k];
            }
        }
        radial_params[i][4] = smallestRoot(temp);
    }

    for (int i = 0; i < 4; i++)
        this->mp.rad[i] = radial_params[0][i];

    this->mp.rad[5] = radial_params[0][4];
    this->mp.rad[4] = ((double)(param.cropWidth < param.cropHeight ? param.cropWidth : param.cropHeight)) / 2.0;
}

bool Remap::remapImage(double & x_dest, double & y_dest, double x_src, double y_src)
{
    x_src -= srcTX - 0.5;
    y_src -= srcTY - 0.5;

    //rotate_erect  中心归一化
    x_dest = x_src + this->mp.rot[1];

    while (x_dest < -this->mp.rot[0])
        x_dest += 2 * this->mp.rot[0];

    while (x_dest >   this->mp.rot[0])
        x_dest -= 2 * this->mp.rot[0];

    y_dest = y_src;

    x_src = x_dest;
    y_src = y_dest;

    //sphere_tp_erect 球面坐标转化为现实坐标
    register double phi, theta, r, s;
    double v[3];
    phi = x_src / mp.distance; //
    theta = -y_src / mp.distance + PI / 2; //
    if (theta < 0)
    {
        theta = -theta;
        phi += PI;
    }
    if (theta > PI)
    {
        theta = PI - (theta - PI);
        phi += PI;
    }
    /*s = sin( theta );
    v[0] =  s * sin( phi );	//  y' -> x
    v[1] =  cos( theta );				//  z' -> y
    r = sqrt( v[1]*v[1] + v[0]*v[0]);
    theta = mp.distance * atan2( r , s * cos( phi ) );
    x_dest =  theta * v[0] / r;
    y_dest =  theta * v[1] / r;
    x_src = x_dest ;
    y_src = y_dest ;

    r 		= sqrt( x_src * x_src + y_src * y_src );
    theta 	= r / mp.distance;
    if( r == 0.0 )
    s = 0.0;
    else
    s = sin( theta ) / r;

    v[0] =  s * x_src ;
    v[1] =  s * y_src ;
    //theta = atan2( r , s * cos( phi ) );
    v[2] =  cos( theta );*/

    v[0] = sin(theta) * sin(phi);
    v[1] = cos(theta);
    v[2] = sin(theta) * cos(phi);

    //摄像机外参
    //matrix_inv_mult( (double(*)[3]) ((void**)params)[0], v );
    register int i;
    register double v0 = v[0];
    register double v1 = v[1];
    register double v2 = v[2];

    for (i = 0; i<3; i++)
    {
        v[i] = mp.mt[0][i] * v0 + mp.mt[1][i] * v1 + mp.mt[2][i] * v2;
    }

    r = sqrt(v[0] * v[0] + v[1] * v[1]);
    if (r == 0.0)
        theta = 0.0;
    else
        theta = mp.distance * atan2(r, v[2]) / r;
    x_dest = theta * v[0];
    y_dest = theta * v[1];
    x_src = x_dest;
    y_src = y_dest;

    if (srcImageType == PTImageTypeRectlinear)                                    // rectilinear image
    {
        //SetDesc(m_stack[i],   rect_sphere_tp,         &(m_mp.distance) ); i++; // Convert rectilinear to spherical
        register double rho, theta, r;
        r = sqrt(x_src * x_src + y_src * y_src);
        theta = r / mp.distance;

        if (theta >= PI / 2.0)
            rho = 1.6e16;
        else if (theta == 0.0)
            rho = 1.0;
        else
            rho = tan(theta) / theta;
        x_dest = rho * x_src;
        y_dest = rho * y_src;
        x_src = x_dest;
        y_src = y_dest;
    }

    //摄像机内参
    //SetDesc(  stack[i],   resize,                 mp.scale       ); i++; // Scale image
    x_dest = x_src * mp.scale[0];
    y_dest = y_src * mp.scale[1];

    x_src = x_dest;
    y_src = y_dest;

    register double rt, scale;

    rt = (sqrt(x_src*x_src + y_src*y_src)) / mp.rad[4];
    if (rt < mp.rad[5])
    {
        scale = ((mp.rad[3] * rt + mp.rad[2]) * rt +
            mp.rad[1]) * rt + mp.rad[0];
    }
    else
        scale = 1000.0;

    x_dest = x_src * scale;
    y_dest = y_src * scale;

    x_src = x_dest;
    y_src = y_dest;

    //摄像机水平竖直矫正
    if (mp.vertical != 0.0)
    {
        //SetDesc(stack[i],   vert,                   &(mp.vertical));   i++;
        x_dest = x_src;
        y_dest = y_src + mp.vertical;
        x_src = x_dest;
        y_src = y_dest;
    }

    if (mp.horizontal != 0.0)
    {
        //SetDesc(stack[i],   horiz,                  &(mp.horizontal)); i++;
        x_dest = x_src + mp.horizontal;
        y_dest = y_src;
        x_src = x_dest;
        y_src = y_dest;
    }

    if (mp.shear[0] != 0 || mp.shear[1] != 0)
    {
        //SetDesc( stack[i],  shear,                  mp.shear       ); i++;
        x_dest = x_src + mp.shear[0] * y_src;
        y_dest = y_src + mp.shear[1] * x_src;
    }

    x_dest += destTX - 0.5;
    y_dest += destTY - 0.5;

    //x_dest += this->mp.rot[1];

    return true;
}

bool Remap::inverseRemapImage(double x_dest, double y_dest, double & x_src, double & y_src)
{
    x_dest -= srcTX - 0.5;
    y_dest -= srcTY - 0.5;

    //shear correction
    if (mp.shear[0] != 0 || mp.shear[1] != 0)
    {
        x_src = x_dest + mp.shear[0] * y_dest;
        y_src = y_dest + mp.shear[1] * x_dest;
    }

    // horizontal correction

    if (mp.horizontal != 0.0)
    {
        x_src = x_dest + mp.horizontal;
        y_src = y_dest;
        x_dest = x_src;
        y_dest = y_src;
    }
    //vertical correction
    if (mp.vertical != 0.0)
    {
        x_src = x_dest;
        y_src = y_dest + mp.vertical;
    }

    //inverse radial correction
    x_dest = x_src;
    y_dest = y_src;

    register double rs, rd, f, scale;
    int iter = 0;
    rd = (sqrt(x_dest*x_dest + y_dest*y_dest)) / (double)mp.rad[4]; // Normalized 

    rs = rd;
    f = (((mp.rad[3] * rs + mp.rad[2]) * rs + mp.rad[1]) * rs + mp.rad[0]) * rs;

    while (abs(f - rd) > R_EPS && iter++ < MAXITER)
    {
        rs = rs - (f - rd) / (((4 * mp.rad[3] * rs + 3 * mp.rad[2]) * rs +
            2 * mp.rad[1]) * rs + mp.rad[0]);

        f = (((mp.rad[3] * rs + mp.rad[2]) * rs + mp.rad[1]) * rs + mp.rad[0]) * rs;
    }

    scale = rs / rd;
    //	printf("scale = %lg iter = %d\n", scale,iter);	
    x_src = x_dest * scale;
    y_src = y_dest * scale;

    // scale 
    x_src = x_src * mp.scale[0]; //3.0887488917637373
    y_src = y_src *mp.scale[1]; //-227.52315797376562

    if (srcImageType == PTImageTypeRectlinear)                                    // rectilinear image
    {
        register double  theta, r;
        r = sqrt(x_src*x_src + y_src*y_src) / mp.distance;
        if (r == 0.0)
            theta = 1.0;
        else
            theta = atan(r) / r;

        x_dest = theta * x_dest;
        y_dest = theta * y_dest;
        x_src = x_dest;
        y_src = y_dest;

    }

    //Perspective Control spherical Image
    x_dest = x_src;
    y_dest = y_src;

    register double phi, theta, r, s;
    double v[3];

    r = sqrt(x_dest * x_dest + y_dest * y_dest);
    theta = r / mp.distance;
    if (r == 0.0)
        s = 0.0;
    else
        s = sin(theta) / r;

    v[0] = s * x_dest;
    v[1] = s * y_dest;
    v[2] = cos(theta);

    register double v0 = v[0];
    register double v1 = v[1];
    register double v2 = v[2];

    for (int i = 0; i<3; i++)
    {
        v[i] = mp.mt[0][i] * v0 + mp.mt[1][i] * v1 + mp.mt[2][i] * v2;
    }

    r = sqrt(v[0] * v[0] + v[1] * v[1]);
    if (r == 0.0)
        theta = 0.0;
    else
        theta = mp.distance * atan2(r, v[2]) / r;
    x_src = theta * v[0];
    y_src = theta * v[1];

    //Convert equirectangular to spherical
    x_dest = x_src;
    y_dest = y_src;

    r = sqrt(x_dest * x_dest + y_dest * y_dest);
    theta = r / mp.distance;
    if (theta == 0.0)
        s = 1.0 / mp.distance;
    else
        s = sin(theta) / r;

    v[1] = s * x_dest;
    v[0] = cos(theta);


    x_src = mp.distance * atan2(v[1], v[0]);
    y_src = mp.distance * atan(s * y_dest / sqrt(v[0] * v[0] + v[1] * v[1]));


    x_dest = x_src;
    y_dest = y_src;


    //rotation correction
    //x_src = x_dest; //+ mp.rot[1];
    x_src = x_dest + mp.rot[1];

    while (x_src < -mp.rot[0])
        x_src += 2 * mp.rot[0];

    while (x_src >  mp.rot[0])
        x_src -= 2 * mp.rot[0];

    y_src = y_dest;

    x_src += destTX - 0.5;
    y_src += destTY - 0.5;

    return true;
}