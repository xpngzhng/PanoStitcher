#include "RunTimeObjects.h"
#include "IntelOpenCLInterface.h"

void ioclSetZero(iocl::UMat& mat, OpenCLBasic& ocl, OpenCLProgramOneKernel& setZeroKern)
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

void setZero(iocl::UMat& mat)
{
    CV_Assert(mat.data && iocl::ocl && iocl::ocl->context && iocl::ocl->queue && iocl::setZero && iocl::setZero->kernel);

    cl_int err = CL_SUCCESS;

    int elemSize = mat.elemSize();
    cl_kernel kernel = iocl::setZero->kernel;
    cl_command_queue queue = iocl::ocl->queue;

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

void convert32SC4To8UC4(const iocl::UMat& src, iocl::UMat& dst)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(src.data && src.type == CV_32SC4);

    dst.create(src.size(), CV_8UC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::convert32SC4To8UC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &dst.step);
    clSetKernelArg(kernel, 4, sizeof(int), &src.rows);
    clSetKernelArg(kernel, 5, sizeof(int), &src.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void convert32FC4To8UC4(const iocl::UMat& src, iocl::UMat& dst)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(src.data && src.type == CV_32FC4);

    dst.create(src.size(), CV_8UC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::convert32FC4To8UC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &dst.step);
    clSetKernelArg(kernel, 4, sizeof(int), &src.rows);
    clSetKernelArg(kernel, 5, sizeof(int), &src.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void setZero8UC4Mask8UC1(iocl::UMat& mat, const iocl::UMat& mask)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(mat.data && mat.type == CV_8UC4 && mask.data && mask.type == CV_8UC1 &&
        mat.size() == mask.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::setZero8UC4Mask8UC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &mask.mem);
    clSetKernelArg(kernel, 5, sizeof(int), &mask.step);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void setVal16SC1(iocl::UMat& mat, short val)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(mat.data && mat.type == CV_16SC1);
    
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::setVal16SC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    clSetKernelArg(kernel, 4, sizeof(short), &val);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void setVal16SC1Mask8UC1(iocl::UMat& mat, short val, const iocl::UMat& mask)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(mat.data && mat.type == CV_16SC1 && mask.data && mask.type == CV_8UC1 &&
        mat.size() == mask.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::setVal16SC1Mask8UC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    clSetKernelArg(kernel, 3, sizeof(int), &mat.step);
    clSetKernelArg(kernel, 4, sizeof(short), &val);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &mask.mem);
    clSetKernelArg(kernel, 6, sizeof(int), &mask.step);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void scaledSet16SC1Mask32SC1(iocl::UMat& image, short val, const iocl::UMat& mask)
{
    CV_Assert(image.data && image.type == CV_16SC1 && mask.data && mask.type == CV_32SC1 && image.size() == mask.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::scaledSet16SC1Mask32SC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &image.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &image.rows);
    clSetKernelArg(kernel, 2, sizeof(int), &image.cols);
    clSetKernelArg(kernel, 3, sizeof(int), &image.step);
    clSetKernelArg(kernel, 4, sizeof(short), &val);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &mask.mem);
    clSetKernelArg(kernel, 6, sizeof(int), &mask.step);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(image.cols, 16), (size_t)round_up_aligned(image.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void subtract16SC4(const iocl::UMat& a, const iocl::UMat& b, iocl::UMat& c)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(a.data && a.type == CV_16SC4 && b.data && b.type == CV_16SC4 && a.size() == b.size());

    c.create(a.size(), CV_16SC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::subtract16SC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &a.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &b.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &b.step);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &c.mem);
    clSetKernelArg(kernel, 5, sizeof(int), &c.step);
    clSetKernelArg(kernel, 6, sizeof(int), &a.rows);
    clSetKernelArg(kernel, 7, sizeof(int), &a.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(a.cols, 16), (size_t)round_up_aligned(a.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void add32SC4(const iocl::UMat& a, const iocl::UMat& b, iocl::UMat& c)
{
    CV_Assert(iocl::ocl && iocl::ocl->context);
    CV_Assert(a.data && a.type == CV_32SC4 && b.data && b.type == CV_32SC4 && a.size() == b.size());

    c.create(a.size(), CV_32SC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::add32SC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &a.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &b.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &b.step);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &c.mem);
    clSetKernelArg(kernel, 5, sizeof(int), &c.step);
    clSetKernelArg(kernel, 6, sizeof(int), &a.rows);
    clSetKernelArg(kernel, 7, sizeof(int), &a.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(a.cols, 16), (size_t)round_up_aligned(a.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void accumulate16SC1To32SC1(const iocl::UMat& src, iocl::UMat& dst)
{
    CV_Assert(src.data && src.type == CV_16SC1 && dst.data && dst.type == CV_32SC1 && src.size() == dst.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::accumulate16SC1To32SC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &dst.step);
    clSetKernelArg(kernel, 4, sizeof(int), &src.rows);
    clSetKernelArg(kernel, 5, sizeof(int), &src.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void accumulate16SC4To32SC4(const iocl::UMat& src, const iocl::UMat& weight, iocl::UMat& dst)
{
    CV_Assert(src.data && src.type == CV_16SC4 && weight.data && weight.type == CV_16SC1 &&
        dst.data && dst.type == CV_32SC4 && src.size() == weight.size() && src.size() == dst.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::accumulate16SC4To32SC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &src.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &weight.step);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst.mem);
    clSetKernelArg(kernel, 5, sizeof(int), &dst.step);
    clSetKernelArg(kernel, 6, sizeof(int), &src.rows);
    clSetKernelArg(kernel, 7, sizeof(int), &src.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 16), (size_t)round_up_aligned(src.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void normalize32SC4(iocl::UMat& mat)
{
    CV_Assert(mat.data && mat.type == CV_32SC4);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::normalizeByShift32SC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &mat.rows);
    clSetKernelArg(kernel, 2, sizeof(int), &mat.cols);
    clSetKernelArg(kernel, 3, sizeof(int), &mat.step);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}

void normalize32SC4(iocl::UMat& mat, const iocl::UMat& weight)
{
    CV_Assert(mat.data && mat.type == CV_32SC4 && weight.data && weight.type == CV_32SC1 &&
        mat.size() == weight.size());

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = iocl::normalizeByDivide32SC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat.mem);
    clSetKernelArg(kernel, 1, sizeof(int), &mat.step);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight.mem);
    clSetKernelArg(kernel, 3, sizeof(int), &weight.step);
    clSetKernelArg(kernel, 4, sizeof(int), &mat.rows);
    clSetKernelArg(kernel, 5, sizeof(int), &mat.cols);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(mat.cols, 16), (size_t)round_up_aligned(mat.rows, 16) };
    size_t localWorkSize[2] = { 16, 16 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
}