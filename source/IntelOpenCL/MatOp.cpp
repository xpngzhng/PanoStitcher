#include "RunTimeObjects.h"
#include "IntelOpenCLInterface.h"

void ioclSetZero(IOclMat& mat, OpenCLBasic& ocl, OpenCLProgramOneKernel& setZeroKern)
{
    CV_Assert(mat.data && ocl.queue && setZeroKern.kernel);
    
    cl_int err = CL_SUCCESS;

    int elemSize = mat.elemSize();

    err = clSetKernelArg(setZeroKern.kernel, 0, sizeof(cl_mem), (void *)&mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(setZeroKern.kernel, 1, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(setZeroKern.kernel, 2, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(setZeroKern.kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(setZeroKern.kernel, 4, sizeof(int), &elemSize);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(ocl.queue, setZeroKern.kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);
}

void ioclSetZero(IOclMat& mat)
{
    CV_Assert(mat.data && iocl::ocl && iocl::ocl->queue && iocl::setZero && iocl::setZero->kernel);

    cl_int err = CL_SUCCESS;

    int elemSize = mat.elemSize();

    err = clSetKernelArg(iocl::setZero->kernel, 0, sizeof(cl_mem), (void *)&mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(iocl::setZero->kernel, 1, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(iocl::setZero->kernel, 2, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(iocl::setZero->kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(iocl::setZero->kernel, 4, sizeof(int), &elemSize);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(iocl::ocl->queue, iocl::setZero->kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(iocl::ocl->queue);
    SAMPLE_CHECK_ERRORS(err);
}