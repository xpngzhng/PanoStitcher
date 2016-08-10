#include "RunTimeObjects.h"
#include "OpenCLAccel/oclobject.hpp"
#include "OpenCLAccel/CompileControl.h"
#include "OpenCLAccel/ProgramSourceStrings.h"

namespace iocl
{

OpenCLBasic* ocl = 0;

OpenCLProgramOneKernel* convert32SC4To8UC4 = 0;
OpenCLProgramOneKernel* convert32FC4To8UC4 = 0;

OpenCLProgramOneKernel* setZero = 0;
OpenCLProgramOneKernel* setZero8UC4Mask8UC1 = 0;
OpenCLProgramOneKernel* setVal16SC1 = 0;
OpenCLProgramOneKernel* setVal16SC1Mask8UC1 = 0;
OpenCLProgramOneKernel* scaledSet16SC1Mask32SC1 = 0;
OpenCLProgramOneKernel* subtract16SC4 = 0;
OpenCLProgramOneKernel* add32SC4 = 0;
OpenCLProgramOneKernel* accumulate16SC1To32SC1 = 0;
OpenCLProgramOneKernel* accumulate16SC4To32SC4 = 0;
OpenCLProgramOneKernel* normalizeByShift32SC4 = 0;
OpenCLProgramOneKernel* normalizeByDivide32SC4 = 0;

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

OpenCLProgramOneKernel* pyrUp8UC4To8UC4 = 0;
OpenCLProgramOneKernel* pyrUp16SC4To16SC4 = 0;
OpenCLProgramOneKernel* pyrUp32SC4To32SC4 = 0;

static int hasInit = 0;
bool init()
{
    if (hasInit)
        return true;

    try
    {
        ocl = new OpenCLBasic("Intel", "GPU");

        convert32SC4To8UC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "convert32SC4To8UC4", APPEND_BUILD_OPTION);
        convert32FC4To8UC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "convert32FC4To8UC4", APPEND_BUILD_OPTION);

        setZero =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "setZeroKernel", APPEND_BUILD_OPTION);
        setZero8UC4Mask8UC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "setZero8UC4Mask8UC1", APPEND_BUILD_OPTION);
        setVal16SC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "setVal16SC1", APPEND_BUILD_OPTION);
        setVal16SC1Mask8UC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "setVal16SC1Mask8UC1", APPEND_BUILD_OPTION);
        scaledSet16SC1Mask32SC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "scaledSet16SC1Mask32SC1", APPEND_BUILD_OPTION);
        subtract16SC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "subtract16SC4", APPEND_BUILD_OPTION);
        add32SC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "add32SC4", APPEND_BUILD_OPTION);
        accumulate16SC1To32SC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "accumulate16SC1To32SC1", APPEND_BUILD_OPTION);
        accumulate16SC4To32SC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "accumulate16SC4To32SC4", APPEND_BUILD_OPTION);
        normalizeByShift32SC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "normalizeByShift32SC4", APPEND_BUILD_OPTION);
        normalizeByDivide32SC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"MatOp.txt"), PROG_STRING(sourceMatOp),
            "normalizeByDivide32SC4", APPEND_BUILD_OPTION);

        reproject =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"ReprojectLinearTemplate.txt"),
            PROG_STRING(sourceReprojectLinearTemplate), "reprojectLinearKernel",
            "-D DST_TYPE=uchar"APPEND_BUILD_OPTION);
        reprojectTo16S =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"ReprojectLinearTemplate.txt"),
            PROG_STRING(sourceReprojectLinearTemplate), "reprojectLinearKernel",
            "-D DST_TYPE=short"APPEND_BUILD_OPTION);
        reprojectWeightedAccumulateTo32F =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"ReprojectWeightedAccumulate.txt"),
            PROG_STRING(sourceReprojectWeightedAccumulate), "reprojectWeightedAccumulateTo32FKernel",
            APPEND_BUILD_OPTION);

        pyrDown8UC1To8UC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownTemplate.txt"),
            PROG_STRING(sourcePyramidDownTemplate), "pyrDownKernel",
            "-D SRC_TYPE=uchar -D DST_TYPE=uchar -D WORK_TYPE=int -D VERT_REFLECT -D HORI_WRAP -D NORMALIZE "
            "-D CONVERT_WORK_TYPE=convert_int -D CONVERT_DST_TYPE=convert_uchar"APPEND_BUILD_OPTION);
        pyrDown8UC4To8UC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownTemplateFP.txt"),
            PROG_STRING(sourcePyramidDownTemplateFP), "pyrDownKernel",
            "-D SRC_TYPE=uchar4 -D DST_TYPE=uchar4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP -D NORMALIZE "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_uchar4_sat_rtz"APPEND_BUILD_OPTION);
        pyrDown8UC4To32SC4 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownTemplateFP.txt"),
            PROG_STRING(sourcePyramidDownTemplateFP), "pyrDownKernel",
            "-D SRC_TYPE=uchar4 -D DST_TYPE=int4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_int4_sat_rtz"APPEND_BUILD_OPTION);

        pyrDown32FC1 =
            new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownPureFP.txt"),
            PROG_STRING(sourcePyramidDownPureFP), "pyrDownKernel",
            "-D TYPE=float -D VERT_REFLECT -D HORI_WRAP"APPEND_BUILD_OPTION);
        pyrDown32FC4 = new OpenCLProgramOneKernel(*ocl, L"PyramidDownPureFP.txt", "", "pyrDownKernel",
            "-D TYPE=float4 -D VERT_REFLECT -D HORI_WRAP"APPEND_BUILD_OPTION);

        pyrDown16SC1To16SC1 = new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownTemplateFP.txt"),
            PROG_STRING(sourcePyramidDownTemplateFP), "pyrDownKernel",
            "-D SRC_TYPE=short -D DST_TYPE=short -D WORK_TYPE=float -D VERT_REFLECT -D HORI_WRAP -D NORMALIZE "
            "-D CONVERT_WORK_TYPE=convert_float -D CONVERT_DST_TYPE=convert_short_sat_rtz"APPEND_BUILD_OPTION);
        pyrDown16SC1To32SC1 = new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownTemplateFP.txt"),
            PROG_STRING(sourcePyramidDownTemplateFP), "pyrDownKernel",
            "-D SRC_TYPE=short -D DST_TYPE=int -D WORK_TYPE=float -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float -D CONVERT_DST_TYPE=convert_int_sat_rtz"APPEND_BUILD_OPTION);

        pyrDown16SC4ScaleTo16SC4 = new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidDownTemplateFP.txt"),
            PROG_STRING(sourcePyramidDownTemplateFP), "pyrDownKernel",
            "-D SRC_TYPE=short4 -D DST_TYPE=short4 -D WORK_TYPE=float4 -D SCALE_TYPE=int -D SCALE -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_short4_sat_rtz"APPEND_BUILD_OPTION);

        pyrUp8UC4To8UC4 = new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidUpTemplateFP.txt"),
            PROG_STRING(sourcePyramidUpTemplateFP), "pyrUpKernel",
            "-D SRC_TYPE=uchar4 -D DST_TYPE=uchar4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_uchar4_sat_rtz"APPEND_BUILD_OPTION);
        pyrUp16SC4To16SC4 = new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidUpTemplateFP.txt"),
            PROG_STRING(sourcePyramidUpTemplateFP), "pyrUpKernel",
            "-D SRC_TYPE=short4 -D DST_TYPE=short4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_short4_sat_rtz"APPEND_BUILD_OPTION);
        pyrUp32SC4To32SC4 = new OpenCLProgramOneKernel(*ocl, PROG_FILE_NAME(L"PyramidUpTemplateFP.txt"),
            PROG_STRING(sourcePyramidUpTemplateFP), "pyrUpKernel",
            "-D SRC_TYPE=int4 -D DST_TYPE=int4 -D WORK_TYPE=float4 -D VERT_REFLECT -D HORI_WRAP "
            "-D CONVERT_WORK_TYPE=convert_float4 -D CONVERT_DST_TYPE=convert_int4_sat_rtz"APPEND_BUILD_OPTION);
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