#pragma once

#include "opencv2/core.hpp"
#include "CL/cl.h"
#include <memory>

//struct IOclMat
//{
//    enum { ptrAlignSize = 128 };
//    enum { stepAlignSize = 128 };
//
//    IOclMat()
//    {
//        data = 0;
//        rows = 0;
//        cols = 0;
//        step = 0;
//        type = 0;
//
//        ctx = 0;
//        mem = 0;
//    }
//
//    IOclMat(int rows, int cols, int type, cl_context ctx)
//    {
//        create(rows, cols, type, ctx);
//    }
//
//    IOclMat(const cv::Size& size, int type, cl_context ctx)
//    {
//        create(size, type, ctx);
//    }
//
//    IOclMat(int rows, int cols, int type, unsigned char* data, int step, cl_context ctx)
//    {
//        create(rows, cols, type, data, step, ctx);
//    }
//
//    IOclMat(const cv::Size& size, int type, unsigned char* data, int step, cl_context ctx)
//    {
//        create(size, type, data, step, ctx);
//    }
//
//    void clear()
//    {
//        data = 0;
//        rows = 0;
//        cols = 0;
//        step = 0;
//        type = 0;
//        sdata.reset();
//
//        ctx = 0;
//        mem = 0;
//        smem.reset();
//    }
//
//    void release()
//    {
//        clear();
//    }
//
//    void create(int rows_, int cols_, int type_, cl_context ctx_)
//    {
//        if (rows == rows_ && cols == cols_ && type == (type_& CV_MAT_TYPE_MASK) && ctx == ctx_)
//            return;
//
//        if (rows_ <= 0 || cols_ <= 0 || !ctx_)
//        {
//            clear();
//            return;
//        }
//
//        int dataChanged = 0;
//        if (rows != rows_ || cols != cols_ || type != (type_& CV_MAT_TYPE_MASK))
//        {
//            dataChanged = 1;
//            sdata.reset();
//            rows = rows_;
//            cols = cols_;
//            type = type_ & CV_MAT_TYPE_MASK;
//            int channels = CV_MAT_CN(type);
//            int elemSize1 = 1 << (CV_MAT_DEPTH(type) / 2);
//            step = (elemSize1 * channels * cols + stepAlignSize - 1) / stepAlignSize * stepAlignSize;
//            data = (unsigned char*)_aligned_malloc(step * rows, ptrAlignSize);
//            sdata.reset(data, _aligned_free);
//        }
//
//        if (dataChanged || ctx != ctx_)
//        {
//            smem.reset();
//            ctx = ctx_;
//            int err = 0;
//            mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * step, data, &err);
//            smem.reset(mem, clReleaseMemObject);
//            if (err)
//            {
//                clear();
//            }
//        }
//    }
//
//    void create(const cv::Size& size_, int type_, cl_context ctx_)
//    {
//        create(size_.height, size_.width, type_, ctx_);
//    }
//
//    void create(int rows_, int cols_, int type_, unsigned char* data_, int step_, cl_context ctx_)
//    {
//        clear();
//
//        if (rows_ <= 0 || cols_ <= 0 || !ctx_)
//            return;
//
//        rows = rows_;
//        cols = cols_;
//        step = step_;
//        type = type_ & CV_MAT_TYPE_MASK;
//        data = data_;
//        ctx = ctx_;
//        int err = 0;
//        mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * step, data, &err);
//        smem.reset(mem, clReleaseMemObject);
//        if (err)
//            clear();
//    }
//
//    void create(const cv::Size& size_, int type_, unsigned char* data_, int step_, cl_context ctx_)
//    {
//        create(size_.height, size_.width, type_, data_, step_, ctx_);
//    }
//
//    void copyTo(IOclMat& other) const
//    {
//        other.create(size(), type, ctx);
//        cv::Mat src = toOpenCVMat();
//        cv::Mat dst = other.toOpenCVMat();
//        src.copyTo(dst);
//    }
//
//    IOclMat clone() const
//    {
//        IOclMat other;
//        copyTo(other);
//        return other;
//    }
//
//    cv::Mat toOpenCVMat() const
//    {
//        if (data)
//            return cv::Mat(rows, cols, type, data, step);
//        else
//            return cv::Mat();
//    }
//
//    void upload(const cv::Mat& mat, cl_context ctx_)
//    {
//        CV_Assert(mat.data);
//        create(mat.size(), mat.type(), ctx_);
//        cv::Mat header = toOpenCVMat();
//        mat.copyTo(header);
//    }
//
//    void download(cv::Mat& mat) const
//    {
//        CV_Assert(data);
//        mat.create(size(), type);
//        cv::Mat header = toOpenCVMat();
//        header.copyTo(mat);
//    }
//
//    void setZero()
//    {
//        if (data)
//        {
//            int actualLineSize = CV_MAT_CN(type) * (1 << (CV_MAT_DEPTH(type) / 2)) * cols;
//            for (int i = 0; i < rows; i++)
//                memset(data + i * step, 0, actualLineSize);
//        }
//    }
//
//    cv::Size size() const
//    {
//        return cv::Size(cols, rows);
//    }
//
//    int depth() const
//    {
//        return CV_MAT_DEPTH(type);
//    }
//
//    int elemSize() const
//    {
//        return CV_MAT_CN(type) * (1 << (CV_MAT_DEPTH(type) / 2));
//    }
//
//    int elemSize1() const
//    {
//        return (1 << (CV_MAT_DEPTH(type) / 2));
//    }
//
//    int channels() const
//    {
//        if (!data)
//            return 0;
//        return CV_MAT_CN(type);
//    }
//
//    unsigned char* data;
//    int rows, cols;
//    int step;
//    int type;
//    std::shared_ptr<unsigned char> sdata;
//
//    cl_context ctx;
//    cl_mem mem;
//    std::shared_ptr<_cl_mem> smem;
//};

#include "RunTimeObjects.h"
namespace docl
{

struct HostMem
{
    HostMem()
    {
        ctx = 0;
        mem = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
        mapped = 0;
        mappedPtr = 0;
    }

    HostMem(int rows, int cols, int type)
    {
        create(rows, cols, type);
    }

    HostMem(const cv::Size& size, int type)
    {
        create(size, type);
    }

    void clear()
    {
        CV_Assert(mapped == 0);
        mappedPtr = 0;

        ctx = 0;
        mem = 0;
        smem.reset();

        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
    }

    void release()
    {
        clear();
    }

    void create(int rows_, int cols_, int type_)
    {
        if (rows == rows_ && cols == cols_ && type == (type_& CV_MAT_TYPE_MASK))
            return;

        if (rows_ <= 0 || cols_ <= 0)
        {
            clear();
            return;
        }

        CV_Assert(mapped == 0);
        mappedPtr = 0;
        if (rows != rows_ || cols != cols_ || type != (type_& CV_MAT_TYPE_MASK))
        {
            smem.reset();
            rows = rows_;
            cols = cols_;
            type = type_ & CV_MAT_TYPE_MASK;
            CV_Assert(ocl && ocl->context);
            ctx = ocl->context;
            int channels = CV_MAT_CN(type);
            int elemSize1 = 1 << (CV_MAT_DEPTH(type) / 2);
            step = elemSize1 * channels * cols;
            int err = 0;
            mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, rows * step, 0, &err);
            if (err) clear();
            SAMPLE_CHECK_ERRORS(err);
            smem.reset(mem, clReleaseMemObject);
        }
    }

    void create(const cv::Size& size_, int type_)
    {
        create(size_.height, size_.width, type_);
    }

    void copyTo(HostMem& other) const
    {
        other.create(size(), type);
        clEnqueueCopyBuffer(ocl->queue, mem, other.mem, 0, 0, step * rows, 0, 0, 0);
        int err = clFinish(ocl->queue);
        SAMPLE_CHECK_ERRORS(err);
    }

    HostMem clone() const
    {
        HostMem other;
        copyTo(other);
        return other;
    }

    cv::Mat mapToHost()
    {
        CV_Assert(mapped == 0);
        int err = 0;
        mappedPtr = clEnqueueMapBuffer(ocl->queue, mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, step * rows, 0, 0, 0, &err);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl->queue);
        SAMPLE_CHECK_ERRORS(err);
        mapped = 1;
        return cv::Mat(rows, cols, type, mappedPtr, step);
    }

    void unmapFromHost(const cv::Mat& mat)
    {
        CV_Assert(mapped == 1 && mat.data == mappedPtr);
        int err = 0;
        err = clEnqueueUnmapMemObject(ocl->queue, mem, mappedPtr, 0, 0, 0);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl->queue);
        SAMPLE_CHECK_ERRORS(err);
        mapped = 0;
        mappedPtr = 0;
    }

    cv::Size size() const
    {
        return cv::Size(cols, rows);
    }

    int depth() const
    {
        return CV_MAT_DEPTH(type);
    }

    int elemSize() const
    {
        return CV_MAT_CN(type) * (1 << (CV_MAT_DEPTH(type) / 2));
    }

    int elemSize1() const
    {
        return (1 << (CV_MAT_DEPTH(type) / 2));
    }

    int channels() const
    {
        if (!mem)
            return 0;
        return CV_MAT_CN(type);
    }

    int mapped;
    void* mappedPtr;

    int rows, cols;
    int step;
    int type;

    cl_context ctx;
    cl_mem mem;
    std::shared_ptr<_cl_mem> smem;
};

struct GpuMat
{
    GpuMat()
    {
        mem = 0;
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;

        ctx = 0;
    }

    GpuMat(int rows, int cols, int type)
    {
        create(rows, cols, type);
    }

    GpuMat(const cv::Size& size, int type)
    {
        create(size, type);
    }

    GpuMat(int rows, int cols, int type, cl_mem data, int step)
    {
        create(rows, cols, type, data, step);
    }

    GpuMat(const cv::Size& size, int type, cl_mem data, int step)
    {
        create(size.height, size.width, type, data, step);
    }

    void clear()
    {
        mem = 0;
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
        smem.reset();

        ctx = 0;
    }

    void release()
    {
        clear();
    }

    void create(int rows_, int cols_, int type_)
    {
        if (rows == rows_ && cols == cols_ && type == (type_& CV_MAT_TYPE_MASK))
            return;

        if (rows_ <= 0 || cols_ <= 0)
        {
            clear();
            return;
        }

        if (rows != rows_ || cols != cols_ || type != (type_& CV_MAT_TYPE_MASK))
        {
            smem.reset();
            rows = rows_;
            cols = cols_;
            type = type_ & CV_MAT_TYPE_MASK;
            CV_Assert(ocl && ocl->context);
            ctx = ocl->context;
            int channels = CV_MAT_CN(type);
            int elemSize1 = 1 << (CV_MAT_DEPTH(type) / 2);
            step = elemSize1 * channels * cols;
            int err = 0;
            mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, rows * step, 0, &err);
            if (err) clear();
            SAMPLE_CHECK_ERRORS(err);
            data = mem;
            smem.reset(mem, clReleaseMemObject);
        }
    }

    void create(int rows_, int cols_, int type_, cl_mem data_, int step_)
    {
        clear();

        CV_Assert(rows_ > 0 && cols_ > 0 && data_ && step_ > 0);

        rows = rows_;
        cols = cols_;
        step = step_;
        type = type_ & CV_MAT_TYPE_MASK;
        CV_Assert(ocl && ocl->context);
        ctx = ocl->context;
        mem = data_;
        data = data_;
    }

    void create(const cv::Size& size_, int type_)
    {
        create(size_.height, size_.width, type_);
    }

    void copyTo(GpuMat& other) const
    {
        other.create(size(), type);
        int err = 0;
        err = clEnqueueCopyBuffer(ocl->queue, mem, other.mem, 0, 0, step * rows, 0, 0, 0);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl->queue);
        SAMPLE_CHECK_ERRORS(err);
    }

    GpuMat clone() const
    {
        GpuMat other;
        copyTo(other);
        return other;
    }

    void upload(const HostMem& mat)
    {
        CV_Assert(mat.mem);

        create(mat.rows, mat.cols, mat.type);
        int err = 0;
        err = clEnqueueCopyBuffer(ocl->queue, mat.mem, mem, 0, 0, step * rows, 0, 0, 0);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl->queue);
        SAMPLE_CHECK_ERRORS(err);
    }

    void download(HostMem& mat) const
    {
        CV_Assert(mem);

        mat.create(rows, cols, type);
        int err = 0;
        err = clEnqueueCopyBuffer(ocl->queue, mem, mat.mem, 0, 0, step * rows, 0, 0, 0);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl->queue);
        SAMPLE_CHECK_ERRORS(err);
    }

    void upload(const cv::Mat& mat)
    {
        CV_Assert(mat.data);

        create(mat.rows, mat.cols, mat.type());
        int err = 0;
        if (mat.isContinuous())
            err = clEnqueueWriteBuffer(ocl->queue, mem, CL_TRUE, 0, step * rows, mat.data, 0, 0, 0);
        else
        {
            size_t deviceOrigin[3] = { 0, 0, 0 };
            size_t hostOrigin[3] = { 0, 0, 0 };
            size_t region[3] = { step, rows, 1 };
            err = clEnqueueWriteBufferRect(ocl->queue, mem, CL_TRUE, deviceOrigin, hostOrigin, region,
                step, 1, mat.step, 1, mat.data, 0, 0, 0);
        }
        SAMPLE_CHECK_ERRORS(err);
    }

    void download(cv::Mat& mat) const
    {
        CV_Assert(mem);

        mat.create(rows, cols, type);
        int err = 0;
        if (mat.isContinuous())
            err = clEnqueueReadBuffer(ocl->queue, mem, CL_TRUE, 0, step * rows, mat.data, 0, 0, 0);
        else
        {
            size_t deviceOrigin[3] = { 0, 0, 0 };
            size_t hostOrigin[3] = { 0, 0, 0 };
            size_t region[3] = { step, rows, 1 };
            err = clEnqueueReadBufferRect(ocl->queue, mem, CL_TRUE, deviceOrigin, hostOrigin, region,
                step, 1, mat.step, 1, mat.data, 0, 0, 0);
        }
        SAMPLE_CHECK_ERRORS(err);
    }

    cv::Size size() const
    {
        return cv::Size(cols, rows);
    }

    int depth() const
    {
        return CV_MAT_DEPTH(type);
    }

    int elemSize() const
    {
        return CV_MAT_CN(type) * (1 << (CV_MAT_DEPTH(type) / 2));
    }

    int elemSize1() const
    {
        return (1 << (CV_MAT_DEPTH(type) / 2));
    }

    int channels() const
    {
        if (!mem)
            return 0;
        return CV_MAT_CN(type);
    }

    int rows, cols;
    int step;
    int type;

    cl_context ctx;
    cl_mem data;
    cl_mem mem;
    std::shared_ptr<_cl_mem> smem;
};
}