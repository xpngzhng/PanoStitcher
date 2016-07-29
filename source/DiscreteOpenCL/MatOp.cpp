#include "RunTimeObjects.h"
#include "DiscreteOpenCLInterface.h"

  void setZero(docl::GpuMat& mat)
{
    CV_Assert(mat.data && docl::ocl && docl::ocl->context && docl::ocl->queue && docl::setZero && docl::setZero->kernel);

    cl_int err = CL_SUCCESS;

    int elemSize = mat.elemSize();
    cl_kernel kernel = docl::setZero->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &elemSize);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void convert32SC4To8UC4(const docl::GpuMat& src, docl::GpuMat& dst)
{
    CV_Assert(src.data && src.type == CV_32SC4);

    dst.create(src.size(), CV_8UC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::convert32SC4To8UC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void convert32FC4To8UC4(const docl::GpuMat& src, docl::GpuMat& dst)
{
    CV_Assert(src.data && src.type == CV_32FC4);

    dst.create(src.size(), CV_8UC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::convert32FC4To8UC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void setZero8UC4Mask8UC1(docl::GpuMat& mat, const docl::GpuMat& mask)
{
    CV_Assert(mat.data && mat.type == CV_8UC4 && mask.data && mask.type == CV_8UC1 &&
        mat.size() == mask.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::setZero8UC4Mask8UC1->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &mask.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &mask.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void setVal16SC1(docl::GpuMat& mat, short val)
{
    CV_Assert(mat.data && mat.type == CV_16SC1);
    
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::setVal16SC1->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(short), &val);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void setVal16SC1Mask8UC1(docl::GpuMat& mat, short val, const docl::GpuMat& mask)
{
    CV_Assert(mat.data && mat.type == CV_16SC1 && mask.data && mask.type == CV_8UC1 &&
        mat.size() == mask.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::setVal16SC1Mask8UC1->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(short), &val);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &mask.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &mask.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void scaledSet16SC1Mask32SC1(docl::GpuMat& image, short val, const docl::GpuMat& mask)
{
    CV_Assert(image.data && image.type == CV_16SC1 && mask.data && mask.type == CV_32SC1 && image.size() == mask.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::scaledSet16SC1Mask32SC1->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &image.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &image.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &image.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(short), &val);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &mask.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &mask.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(image.cols, 16), (size_t)round_up_aligned(image.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void subtract16SC4(const docl::GpuMat& a, const docl::GpuMat& b, docl::GpuMat& c)
{
    CV_Assert(a.data && a.type == CV_16SC4 && b.data && b.type == CV_16SC4 && a.size() == b.size());

    c.create(a.size(), CV_16SC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::subtract16SC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &a.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &b.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &c.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &c.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &a.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &a.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(a.cols, 16), (size_t)round_up_aligned(a.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void add32SC4(const docl::GpuMat& a, const docl::GpuMat& b, docl::GpuMat& c)
{
    CV_Assert(a.data && a.type == CV_32SC4 && b.data && b.type == CV_32SC4 && a.size() == b.size());

    c.create(a.size(), CV_32SC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::add32SC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &a.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &b.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &c.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &c.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &a.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &a.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(a.cols, 16), (size_t)round_up_aligned(a.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void accumulate16SC1To32SC1(const docl::GpuMat& src, docl::GpuMat& dst)
{
    CV_Assert(src.data && src.type == CV_16SC1 && dst.data && dst.type == CV_32SC1 && src.size() == dst.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::accumulate16SC1To32SC1->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void accumulate16SC4To32SC4(const docl::GpuMat& src, const docl::GpuMat& weight, docl::GpuMat& dst)
{
    CV_Assert(src.data && src.type == CV_16SC4 && weight.data && weight.type == CV_16SC1 &&
        dst.data && dst.type == CV_32SC4 && src.size() == weight.size() && src.size() == dst.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::accumulate16SC4To32SC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &weight.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void normalize32SC4(docl::GpuMat& mat)
{
    CV_Assert(mat.data && mat.type == CV_32SC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::normalizeByShift32SC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void normalize32SC4(docl::GpuMat& mat, const docl::GpuMat& weight)
{
    CV_Assert(mat.data && mat.type == CV_32SC4 && weight.data && weight.type == CV_32SC1 &&
        mat.size() == weight.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = docl::normalizeByDivide32SC4->kernel;
    cl_command_queue queue = docl::ocl->queue;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &mat.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &weight.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &mat.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &mat.cols);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}