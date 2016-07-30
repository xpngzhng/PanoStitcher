#include "RunTimeObjects.h"
#include "DiscreteOpenCLInterface.h"

void doclReproject(const docl::GpuMat& src, docl::GpuMat& dst, const docl::GpuMat& xmap, const docl::GpuMat& ymap)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        docl::ocl && docl::ocl->context && docl::ocl->queue && 
        docl::reproject && docl::reproject->kernel);

    dst.create(xmap.rows, ymap.cols, CV_8UC4);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = docl::reproject->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void doclReprojectTo16S(const docl::GpuMat& src, docl::GpuMat& dst, const docl::GpuMat& xmap, const docl::GpuMat& ymap)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        docl::ocl && docl::ocl->context && docl::ocl->queue &&
        docl::reproject && docl::reproject->kernel);

    dst.create(xmap.rows, ymap.cols, CV_16SC4);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = docl::reprojectTo16S->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(dst.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void doclReprojectWeightedAccumulateTo32F(const docl::GpuMat& src, docl::GpuMat& dst,
    const docl::GpuMat& xmap, const docl::GpuMat& ymap, const docl::GpuMat& weight)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        dst.data && dst.type == CV_32FC4 && dst.size() == xmap.size() &&
        xmap.size() == weight.size() && weight.type == CV_32FC1 &&
        docl::ocl && docl::ocl->context && docl::ocl->queue &&
        docl::reprojectWeightedAccumulateTo32F && docl::reprojectWeightedAccumulateTo32F->kernel);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = docl::reprojectWeightedAccumulateTo32F->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&weight.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 13, sizeof(int), &weight.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(dst.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void doclReprojectTo16S(const docl::GpuMat& src, docl::GpuMat& dst, const docl::GpuMat& xmap, const docl::GpuMat& ymap,
    OpenCLProgramOneKernel& kern, OpenCLQueue& q)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        kern.kernel && q.queue);

    dst.create(xmap.rows, ymap.cols, CV_16SC4);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = kern.kernel;
    cl_command_queue queue = q.queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(dst.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    //err = clFinish(queue);
    //SAMPLE_CHECK_ERRORS(err);
}

void doclReprojectWeightedAccumulateTo32F(const docl::GpuMat& src, docl::GpuMat& dst,
    const docl::GpuMat& xmap, const docl::GpuMat& ymap, const docl::GpuMat& weight,
    OpenCLProgramOneKernel& kern, OpenCLQueue& q)
{
    CV_Assert(src.data && src.type == CV_8UC4 && xmap.size() == ymap.size() &&
        xmap.data && xmap.type == CV_32FC1 && ymap.data && ymap.type == CV_32FC1 &&
        dst.data && dst.type == CV_32FC4 && dst.size() == xmap.size() &&
        xmap.size() == weight.size() && weight.type == CV_32FC1 &&
        kern.kernel && q.queue);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = kern.kernel;
    cl_command_queue queue = q.queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&xmap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &xmap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&ymap.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 11, sizeof(int), &ymap.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&weight.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 13, sizeof(int), &weight.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(dst.cols, 16), (size_t)round_up_aligned(dst.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
}
