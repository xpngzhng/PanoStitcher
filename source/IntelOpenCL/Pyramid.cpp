#include "RunTimeObjects.h"
#include "IntelOpenCLMat.h"

#define PYR_DOWN_MAIN_BODY \
err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 1, sizeof(int), &src.rows);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 5, sizeof(int), &dst.rows);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 6, sizeof(int), &dst.cols);\
SAMPLE_CHECK_ERRORS(err);\
err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);\
SAMPLE_CHECK_ERRORS(err);\
\
size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 256), (size_t)round_up_aligned(dst.rows, 1) };\
size_t localWorkSize[2] = { 256, 1 };\
size_t offset[2] = { 0, 0 };\
\
err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);\
SAMPLE_CHECK_ERRORS(err);\
err = clFinish(queue);\
SAMPLE_CHECK_ERRORS(err)

void ioclPyramidDown8UC1To8UC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type == CV_8UC1);
    CV_Assert(iocl::ocl && iocl::ocl->context && iocl::ocl->queue && iocl::pyrDown8UC1To8UC1->kernel);
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols + 1) / 2;
        dstSize.height = (src.rows + 1) / 2;
    }

    dst.create(dstSize, CV_8UC1, iocl::ocl->context);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = iocl::pyrDown8UC1To8UC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    /*
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 256), (size_t)round_up_aligned(dst.rows, 1) };
    size_t localWorkSize[2] = { 256, 1 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
    */
    PYR_DOWN_MAIN_BODY;
}

void ioclPyramidDown8UC4To8UC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type == CV_8UC4);
    CV_Assert(iocl::ocl && iocl::ocl->context && iocl::ocl->queue && iocl::pyrDown8UC4To8UC4->kernel);
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols + 1) / 2;
        dstSize.height = (src.rows + 1) / 2;
    }

    dst.create(dstSize, CV_8UC4, iocl::ocl->context);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = iocl::pyrDown8UC4To8UC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    /*
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 256), (size_t)round_up_aligned(dst.rows, 1) };
    size_t localWorkSize[2] = { 256, 1 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
    */
    PYR_DOWN_MAIN_BODY;
}

void ioclPyramidDown8UC4To32SC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type == CV_8UC4);
    CV_Assert(iocl::ocl && iocl::ocl->context && iocl::ocl->queue && iocl::pyrDown8UC4To32SC4->kernel);
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols + 1) / 2;
        dstSize.height = (src.rows + 1) / 2;
    }

    dst.create(dstSize, CV_32SC4, iocl::ocl->context);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = iocl::pyrDown8UC4To32SC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    /*
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 256), (size_t)round_up_aligned(dst.rows, 1) };
    size_t localWorkSize[2] = { 256, 1 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
    */
    PYR_DOWN_MAIN_BODY;
}

void ioclPyramidDown32FC1(const IOclMat& src, IOclMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type == CV_32FC1);
    CV_Assert(iocl::ocl && iocl::ocl->context && iocl::ocl->queue && iocl::pyrDown32FC1->kernel);
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols + 1) / 2;
        dstSize.height = (src.rows + 1) / 2;
    }

    dst.create(dstSize, CV_32FC1, iocl::ocl->context);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = iocl::pyrDown32FC1->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    /*
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 256), (size_t)round_up_aligned(dst.rows, 1) };
    size_t localWorkSize[2] = { 256, 1 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
    */
    PYR_DOWN_MAIN_BODY;
}

void ioclPyramidDown32FC4(const IOclMat& src, IOclMat& dst, cv::Size dstSize)
{
    CV_Assert(src.data && src.type == CV_32FC4);
    CV_Assert(iocl::ocl && iocl::ocl->context && iocl::ocl->queue && iocl::pyrDown32FC4->kernel);
    if (dstSize == cv::Size())
    {
        dstSize.width = (src.cols + 1) / 2;
        dstSize.height = (src.rows + 1) / 2;
    }

    dst.create(dstSize, CV_32FC4, iocl::ocl->context);

    cl_int err = CL_SUCCESS;

    cl_kernel kernel = iocl::pyrDown32FC4->kernel;
    cl_command_queue queue = iocl::ocl->queue;

    /*
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &src.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &src.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&src.step);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&dst.mem);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst.rows);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &dst.cols);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &dst.step);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalWorkSize[2] = { (size_t)round_up_aligned(src.cols, 256), (size_t)round_up_aligned(dst.rows, 1) };
    size_t localWorkSize[2] = { 256, 1 };
    size_t offset[2] = { 0, 0 };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(queue);
    SAMPLE_CHECK_ERRORS(err);
    */
    PYR_DOWN_MAIN_BODY;
}