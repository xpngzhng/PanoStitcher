#include "RunTimeObjects.h"
#include "oclobject.hpp"

namespace iocl
{

OpenCLBasic* ocl = 0;

OpenCLProgramOneKernel* setZero = 0;

OpenCLProgramOneKernel* reproject = 0;
OpenCLProgramOneKernel* reprojectTo16S = 0;
OpenCLProgramOneKernel* reprojectWeightedAccumulateTo32F = 0;

OpenCLProgramOneKernel* pyrDown8UC1To8UC1 = 0;
OpenCLProgramOneKernel* pyrDown8UC4To8UC4 = 0;
OpenCLProgramOneKernel* pyrDown8UC4To32SC4 = 0;

OpenCLProgramOneKernel* pyrDown32FC1 = 0;
OpenCLProgramOneKernel* pyrDown32FC4 = 0;

OpenCLProgramOneKernel* pyrDown16SC1To16SC1 = 0;
OpenCLProgramOneKernel* pyrDown16SC1To32SC1 = 0;

OpenCLProgramOneKernel* pyrDown16SC4ScaleTo16SC4 = 0;

OpenCLProgramOneKernel* pyrUp8UC4To8UC4;
OpenCLProgramOneKernel* pyrUp16SC4To16SC4;
OpenCLProgramOneKernel* pyrUp32SC4To32SC4;

static int hasInit = 0;
bool init()
{
    if (hasInit)
        return true;

    try
    {
        ocl = new OpenCLBasic("Intel", "GPU");

        setZero = new OpenCLProgramOneKernel(*ocl, L"MatOp.txt", "", "setZeroKernel");

        reproject = new OpenCLProgramOneKernel(*ocl, L"ReprojectLinearTemplate.txt", "", "reprojectLinearKernel", "-D DST_TYPE=uchar");
        reprojectTo16S = new OpenCLProgramOneKernel(*ocl, L"ReprojectLinearTemplate.txt", "", "reprojectLinearKernel", "-D DST_TYPE=short");
        reprojectWeightedAccumulateTo32F = new OpenCLProgramOneKernel(*ocl, L"ReprojectWeightedAccumulate.txt", "", "reprojectWeightedAccumulateTo32FKernel");

        pyrDown8UC1To8UC1 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownTemplate.txt", "", "pyrDownKernel",
            "-D SRC_TYPE=uchar -D DST_TYPE=uchar -D WORK_TYPE=int -D VERT_REFLECT -D HORI_WRAP -D NORMALIZE "
            "-D CONVERT_WORK_TYPE=convert_int -D CONVERT_DST_TYPE=convert_uchar");
        pyrDown8UC4To8UC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownTemplateFP.txt", "", "pyrDownKernel",
            "-D SRC_TYPE=uchar4 -D DST_TYPE=uchar4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP -D NORMALIZE "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_uchar4_sat_rtz");
        pyrDown8UC4To32SC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownTemplateFP.txt", "", "pyrDownKernel",
            "-D SRC_TYPE=uchar4 -D DST_TYPE=int4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_int4_sat_rtz");

        pyrDown32FC1 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownPureFP.txt", "", "pyrDownKernel",
            "-D TYPE=float -D VERT_REFLECT -D HORI_WRAP");
        pyrDown32FC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownPureFP.txt", "", "pyrDownKernel",
            "-D TYPE=float4 -D VERT_REFLECT -D HORI_WRAP");

        pyrDown16SC1To16SC1 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownTemplateFP.txt", "", "pyrDownKernel",
            "-D SRC_TYPE=short -D DST_TYPE=short -D WORK_TYPE=float -D VERT_REFLECT -D HORI_WRAP -D NORMALIZE "
            "-D CONVERT_WORK_TYPE=convert_float -D CONVERT_DST_TYPE=convert_short_sat_rtz");
        pyrDown16SC1To32SC1 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownTemplateFP.txt", "", "pyrDownKernel",
            "-D SRC_TYPE=short -D DST_TYPE=int -D WORK_TYPE=float -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float -D CONVERT_DST_TYPE=convert_int_sat_rtz");

        pyrDown16SC4ScaleTo16SC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownTemplateFP.txt", "", "pyrDownKernel",
            "-D SRC_TYPE=short4 -D DST_TYPE=short4 -D WORK_TYPE=float4 -D SCALE_TYPE=int -D SCALE -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_short4_sat_rtz");

#if !PYR_UP_OPENCV
        pyrUp8UC4To8UC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidUpTemplateFP.txt", "", "pyrUpKernel",
            "-D SRC_TYPE=uchar4 -D DST_TYPE=uchar4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_uchar4_sat_rtz");
        pyrUp16SC4To16SC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidUpTemplateFP.txt", "", "pyrUpKernel",
            "-D SRC_TYPE=short4 -D DST_TYPE=short4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_short4_sat_rtz");
        pyrUp32SC4To32SC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidUpTemplateFP.txt", "", "pyrUpKernel",
            "-D SRC_TYPE=int4 -D DST_TYPE=int4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_int4_sat_rtz");
#else
        pyrUp8UC4To8UC4 = new OpenCLProgramOneKernel(*ocl, L"pyr_up.cl", "", "pyrUp",
            "-D Type=uchar4 -D floatType=float4 -D convertToType=convert_uchar4_sat_rte -D convertToFloat=convert_float4");
        pyrUp16SC4To16SC4 = new OpenCLProgramOneKernel(*ocl, L"pyr_up.cl", "", "pyrUp",
            "-D Type=short4 -D floatType=float4 -D convertToType=convert_short4_sat_rte -D convertToFloat=convert_float4");
        pyrUp32SC4To32SC4 = new OpenCLProgramOneKernel(*ocl, L"pyr_up.cl", "", "pyrUp",
            "-D Type=int4 -D floatType=float4 -D convertToType=convert_int4_sat_rte -D convertToFloat=convert_float4");
#endif
    }
    catch (const std::exception& e)
    {
        printf("Error in %s, exception caught, %s\n", __FUNCTION__, e.what());
        return false;
    }

    hasInit = 1;
    return true;
}

}

bool ioclInit()
{
    return iocl::init();
}