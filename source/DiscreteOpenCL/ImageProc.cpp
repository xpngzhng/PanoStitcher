#include "RunTimeObjects.h"
#include "DiscreteOpenCLInterface.h"

void alphaBlend8UC4(docl::GpuMat& target, const docl::GpuMat& blender)
{
    CV_Assert(docl::ocl && docl::ocl->context && docl::ocl->queue);
    CV_Assert(docl::alphaBlend8UC4 && docl::alphaBlend8UC4->kernel);
    CV_Assert(target.data && target.type == CV_8UC4 &&
        blender.data && blender.type == CV_8UC4 && target.size() == blender.size());
    
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::alphaBlend8UC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &target.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &target.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &blender.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &blender.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &target.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &target.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(target.cols, 16), (size_t)round_up_aligned(target.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void cvtBGR32ToYUV420P(const docl::GpuMat& bgr32, docl::GpuMat& y, docl::GpuMat& u, docl::GpuMat& v)
{
    CV_Assert(docl::ocl && docl::ocl->context && docl::ocl->queue);
    CV_Assert(docl::cvtBGR32ToYUV420P && docl::cvtBGR32ToYUV420P->kernel);
    CV_Assert(bgr32.data && bgr32.type == CV_8UC4 && ((bgr32.rows & 1) == 0) && ((bgr32.cols & 1) == 0));

    int rows = bgr32.rows, cols = bgr32.cols;
    y.create(rows, cols, CV_8UC1);
    u.create(rows / 2, cols / 2, CV_8UC1);
    v.create(rows / 2, cols / 2, CV_8UC1);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::cvtBGR32ToYUV420P->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bgr32.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &bgr32.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &y.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &y.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &u.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &u.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &v.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &v.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(int), &y.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &y.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(bgr32.cols, 16), (size_t)round_up_aligned(bgr32.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void cvtBGR32ToNV12(const docl::GpuMat& bgr32, docl::GpuMat& y, docl::GpuMat& uv)
{
    CV_Assert(docl::ocl && docl::ocl->context && docl::ocl->queue);
    CV_Assert(docl::cvtBGR32ToNV12 && docl::cvtBGR32ToNV12->kernel);
    CV_Assert(bgr32.data && bgr32.type == CV_8UC4 && ((bgr32.rows & 1) == 0) && ((bgr32.cols & 1) == 0));

    int rows = bgr32.rows, cols = bgr32.cols;
    y.create(rows, cols, CV_8UC1);
    uv.create(rows / 2, cols, CV_8UC1);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::cvtBGR32ToNV12->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bgr32.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &bgr32.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &y.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &y.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &uv.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &uv.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &y.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &y.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(bgr32.cols, 16), (size_t)round_up_aligned(bgr32.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}