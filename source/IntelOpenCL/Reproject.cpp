#include "IntelOpenCLInterface.h"

void ioclReproject(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    OpenCLBasic& ocl, OpenCLProgramOneKernel& executable)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        ocl.context && ocl.queue && executable.kernel);

    dst.create(xmap.rows, ymap.cols, CV_8UC4, ocl.context);

    cl_int err = CL_SUCCESS;

    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(dst.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);
}

void ioclReprojectAccumulateWeightedTo32F(const IOclMat& src, IOclMat& dst, const IOclMat& xmap, const IOclMat& ymap,
    const IOclMat& weight, OpenCLBasic& ocl, OpenCLProgramOneKernel& executable)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        dst.data && dst.type == CV_32FC4 && dst.size() == xmap.size() &&
        xmap.size() == weight.size() && weight.type == CV_32FC1 &&
        ocl.context && ocl.queue && executable.kernel);

    cl_int err = CL_SUCCESS;

    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 3, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 12, sizeof(cl_mem), (void *)&weight.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 13, sizeof(int), &weight.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(dst.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);
}
