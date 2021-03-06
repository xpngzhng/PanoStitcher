const char* sourceMatOp = R"(

__kernel void convert32SC4To8UC4(__global const unsigned char* srcData, int srcStep, __global unsigned char* dstData, int dstStep,
    int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
	    ((__global uchar4*)(dstData + dstStep * y))[x] = convert_uchar4_sat_rtz(((__global const int4*)(srcData + srcStep * y))[x]);
	}
}

__kernel void convert32FC4To8UC4(__global const unsigned char* srcData, int srcStep, __global unsigned char* dstData, int dstStep,
    int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
	    ((__global uchar4*)(dstData + dstStep * y))[x] = convert_uchar4_sat_rtz(((__global const float4*)(srcData + srcStep * y))[x]);
	}
}

__kernel void setZeroKernel(__global unsigned char* data, int width, int height, int step, int elemSize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    __global unsigned char* ptr = data + y * step + x * elemSize;
    for (int i = 0; i < elemSize; i++)
        ptr[i] = 0;
}

__kernel void setZero8UC4Mask8UC1(__global unsigned char* data, int rows, int cols, int step,
    __global const unsigned char* maskData, int maskStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
	    if ((maskData + maskStep * y)[x])
            ((__global uchar4*)(data + step * y))[x] = (uchar4)0;
    }
}

__kernel void setVal16SC1(__global unsigned char* data, int rows, int cols, int step, short val)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
        ((__global short*)(data + step * y))[x] = val;
    }
}

__kernel void setVal16SC1Mask8UC1(__global unsigned char* data, int rows, int cols, int step, short val,
    __global const unsigned char* maskData, int maskStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
	    if ((maskData + maskStep * y)[x])
            ((__global short*)(data + step * y))[x] = val;
    }
}

__kernel void scaledSet16SC1Mask32SC1(__global unsigned char* imageData, int imageRows, int imageCols, int imageStep,
    short val, __global const unsigned char* maskData, int maskStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < imageCols && y < imageRows)
    {
        ((__global short*)(imageData + imageStep * y))[x] = ((__global const int*)(maskData + maskStep * y))[x] ? val : 0;
    }
}

__kernel void subtract16SC4(__global const unsigned char* aData, int aStep, __global const unsigned char* bData, int bStep,
    __global unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
        //getRowPtr<short4>(cData, cStep, y)[x] = getElem<short4>(aData, aStep, y, x) - getElem<short4>(bData, bStep, y, x);
		((__global short4*)(cData + cStep * y))[x] = ((__global const short4*)(aData + aStep * y))[x] - ((__global const short4*)(bData + bStep * y))[x];
    }
}

__kernel void add32SC4(__global const unsigned char* aData, int aStep, __global const unsigned char* bData, int bStep,
    __global unsigned char* cData, int cStep, int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
        //getRowPtr<int4>(cData, cStep, y)[x] = getElem<int4>(aData, aStep, y, x) + getElem<int4>(bData, bStep, y, x);
		((__global int4*)(cData + cStep * y))[x] = ((__global const int4*)(aData + aStep * y))[x] + ((__global const int4*)(bData + bStep * y))[x];
    }
}

__kernel void accumulate16SC1To32SC1(__global const unsigned char* srcData, int srcStep, __global unsigned char* dstData, int dstStep,
    int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
        //getRowPtr<int>(dstData, dstStep, y)[x] = getElem<int>(dstData, dstStep, y, x) + getElem<short>(srcData, srcStep, y, x);
		((__global int*)(dstData + dstStep * y))[x] = ((__global const int*)(dstData + dstStep * y))[x] + ((__global const short*)(srcData + srcStep * y))[x];
    }
}

__kernel void accumulate16SC4To32SC4(__global const unsigned char* srcData, int srcStep,
    __global const unsigned char* weightData, int weightStep,
    __global unsigned char* dstData, int dstStep, int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
        //getRowPtr<int4>(dstData, dstStep, y)[x] = getRowPtr<int4>(dstData, dstStep, y)[x] + 
        //    getElem<short>(weightData, weightStep, y, x) * getElem<short4>(srcData, srcStep, y, x);
		((__global int4*)(dstData + dstStep * y))[x] = ((__global int4*)(dstData + dstStep * y))[x] +
		    (int)(((__global const short*)(weightData + weightStep * y))[x]) * convert_int4(((__global const short4*)(srcData + srcStep * y))[x]);
    }
}

__kernel void normalizeByShift32SC4(__global unsigned char* imageData, int imageRows, int imageCols, int imageStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < imageCols && y < imageRows)
    {
        //getRowPtr<int4>(imageData, imageStep, y)[x] = (getElem<int4>(imageData, imageStep, y, x) + make_int4(128, 128, 128, 0)) >> 8;
		((__global int4*)(imageData + imageStep * y))[x] = (((__global int4*)(imageData + imageStep * y))[x] + 128) >> 8;
    }
}

__kernel void normalizeByDivide32SC4(__global unsigned char* imageData, int imageStep,
    __global const unsigned char* weightData, int weightStep, int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x < cols && y < rows)
    {
		int w = ((__global const int*)(weightData + weightStep * y))[x];
        if (!w) w++;
		((__global int4*)(imageData + imageStep * y))[x] = ((__global int4*)(imageData + imageStep * y))[x] / w;
    }
}

)";