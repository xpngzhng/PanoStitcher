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