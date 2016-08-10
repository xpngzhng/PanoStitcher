const char* sourcePyramidUpTemplateFP = R"(

int borderInterpolateWrap(int p, int len)
{
    if( (unsigned)p < (unsigned)len )
        ;
    else
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    return p;
}

int borderInterpolateReflect( int p, int len)
{
    if( (unsigned)p < (unsigned)len )
        ;
    else
    {
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p;
            else
                p = len - 1 - (p - len) - 1;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    return p;
}

#ifdef HORI_WRAP
#define horiBorder borderInterpolateWrap
#elif defined HORI_REFLECT
#define horiBorder borderInterpolateReflect
#else
#error No horizontal interpolate method
#endif

#ifdef VERT_WRAP
#define vertBorder borderInterpolateWrap
#elif defined VERT_REFLECT
#define vertBorder borderInterpolateReflect
#else
#error No vertical interpolate method
#endif

#define PYR_UP_BLOCK_WIDTH 16
#define PYR_UP_BLOCK_HEIGHT 16

inline SRC_TYPE getSourceElem(__global const unsigned char* data, int step, int row, int col)
{
    return ((__global SRC_TYPE*)(data + step * row))[col];
}

inline __global DST_TYPE* getDestRowPtr(__global unsigned char* data, int step, int row)
{
    return (__global DST_TYPE*)(data + step * row);
}

__kernel void pyrUpKernel(__global const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    __global unsigned char* dstData, int dstRows, int dstCols, int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
	const int localx = get_local_id(0);
	const int localy = get_local_id(1);

    __local WORK_TYPE s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 2][PYR_UP_BLOCK_WIDTH / 2 + 2];
    __local WORK_TYPE s_dstPatch[PYR_UP_BLOCK_HEIGHT + 4][PYR_UP_BLOCK_WIDTH];

    if ((localx < PYR_UP_BLOCK_WIDTH / 2 + 2) && (localy < PYR_UP_BLOCK_HEIGHT / 2 + 2))
    {
        int srcx = ((get_group_id(0) * get_local_size(0)) / 2 + localx) - 1;
        int srcy = ((get_group_id(1) * get_local_size(1)) / 2 + localy) - 1;

        srcx = ((srcx < 0) || (srcx >= srcCols)) ? horiBorder(srcx, srcCols) : srcx;
        srcy = ((srcy < 0) || (srcy >= srcRows)) ? vertBorder(srcy, srcRows) : srcy;

        s_srcPatch[localy][localx] = CONVERT_WORK_TYPE(getSourceElem(srcData, srcStep, srcy, srcx));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    WORK_TYPE sum = 0;

    const WORK_TYPE evenFlag = (int)((localx & 1) == 0);
    const WORK_TYPE oddFlag  = (int)((localx & 1) != 0);
    const int eveny = ((localy & 1) == 0);
    const int tidx = localx;

    if (eveny)
    {
        sum =       (evenFlag       ) * s_srcPatch[1 + (localy >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 4.0F) * s_srcPatch[1 + (localy >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 6.0F) * s_srcPatch[1 + (localy >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 4.0F) * s_srcPatch[1 + (localy >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag       ) * s_srcPatch[1 + (localy >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + localy][localx] = sum;

    if (localy < 2)
    {
        if (eveny)
        {
            sum =       (evenFlag       ) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4.0F) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6.0F) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4.0F) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag       ) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[localy][localx] = sum;
    }

    if (localy > PYR_UP_BLOCK_HEIGHT - 3)
    {
        if (eveny)
        {
            sum =       (evenFlag       ) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 4.0F) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 6.0F) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 4.0F) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag       ) * s_srcPatch[PYR_UP_BLOCK_HEIGHT / 2 + 1][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[4 + localy][localx] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int tidy = localy;

    sum =              s_dstPatch[2 + tidy - 2][localx];
    sum = sum + 4.0F * s_dstPatch[2 + tidy - 1][localx];
    sum = sum + 6.0F * s_dstPatch[2 + tidy    ][localx];
    sum = sum + 4.0F * s_dstPatch[2 + tidy + 1][localx];
    sum = sum +        s_dstPatch[2 + tidy + 2][localx];

    if (x < dstCols && y < dstRows)
        getDestRowPtr(dstData, dstStep, y)[x] = CONVERT_DST_TYPE((sum + 32.0F) / 64.0F);
}

)";