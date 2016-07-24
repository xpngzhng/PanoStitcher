#include "oclobject.hpp"

namespace iocl
{

OpenCLBasic* ocl = 0;

OpenCLProgramOneKernel* setZero = 0;
OpenCLProgramOneKernel* reproject = 0;
OpenCLProgramOneKernel* reprojectWeightedAccumulateTo32F = 0;

static int hasInit = 0;
bool init()
{
    if (hasInit)
        return true;

    try
    {
        ocl = new OpenCLBasic("Intel", "GPU");

        setZero = new OpenCLProgramOneKernel(*ocl, L"MatOp.txt", "", "setZeroKernel");
        reproject = new OpenCLProgramOneKernel(*ocl, L"Reproject.txt", "", "reprojectLinearKernel");
        reprojectWeightedAccumulateTo32F = new OpenCLProgramOneKernel(*ocl, L"Reproject.txt", "", "reprojectWeightedAccumulateTo32FKernel");
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