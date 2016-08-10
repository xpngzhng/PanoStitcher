#pragma once

#include "RunTimeObjects.h"
#include "opencv2/core.hpp"
#include "CL/cl.h"
#include <memory>

namespace iocl
{

struct UMat
{
    enum { ptrAlignSize = 128 };
    enum { stepAlignSize = 128 };

    UMat()
    {
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;

        ctx = 0;
        mem = 0;
    }

    UMat(int rows, int cols, int type)
    {
        init();
        create(rows, cols, type);
    }

    UMat(const cv::Size& size, int type)
    {
        init();
        create(size, type);
    }

    UMat(int rows, int cols, int type, unsigned char* data, int step)
    {
        init();
        create(rows, cols, type, data, step);
    }

    UMat(const cv::Size& size, int type, unsigned char* data, int step)
    {
        init();
        create(size, type, data, step);
    }

    void init()
    {
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;

        ctx = 0;
        mem = 0;
    }

    void clear()
    {
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
        sdata.reset();

        ctx = 0;
        mem = 0;
        smem.reset();
    }

    void release()
    {
        clear();
    }

    void create(int rows_, int cols_, int type_)
    {
        CV_Assert(iocl::ocl && iocl::ocl->context);
        ctx = iocl::ocl->context;

        if (rows == rows_ && cols == cols_ && type == (type_& CV_MAT_TYPE_MASK))
            return;

        if (rows_ <= 0 || cols_ <= 0)
        {
            clear();
            return;
        }

        int dataChanged = 0;
        if (rows != rows_ || cols != cols_ || type != (type_& CV_MAT_TYPE_MASK))
        {
            dataChanged = 1;
            sdata.reset();
            rows = rows_;
            cols = cols_;
            type = type_ & CV_MAT_TYPE_MASK;
            int channels = CV_MAT_CN(type);
            int elemSize1 = 1 << (CV_MAT_DEPTH(type) / 2);
            step = (elemSize1 * channels * cols + stepAlignSize - 1) / stepAlignSize * stepAlignSize;
            data = (unsigned char*)_aligned_malloc(step * rows, ptrAlignSize);
            sdata.reset(data, _aligned_free);
        }

        if (dataChanged)
        {
            smem.reset();
            int err = 0;
            mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * step, data, &err);
            smem.reset(mem, clReleaseMemObject);
            if (err)
            {
                clear();
            }
        }
    }

    void create(const cv::Size& size_, int type_)
    {
        create(size_.height, size_.width, type_);
    }

    void create(int rows_, int cols_, int type_, unsigned char* data_, int step_)
    {
        clear();

        if (rows_ <= 0 || cols_ <= 0)
            return;

        CV_Assert(iocl::ocl && iocl::ocl->context);
        CV_Assert((((long long int)data_) & (ptrAlignSize - 1)) == 0);
        rows = rows_;
        cols = cols_;
        step = step_;
        type = type_ & CV_MAT_TYPE_MASK;
        data = data_;
        ctx = iocl::ocl->context;
        int err = 0;
        mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * step, data, &err);
        smem.reset(mem, clReleaseMemObject);
        if (err)
            clear();
    }

    void create(const cv::Size& size_, int type_, unsigned char* data_, int step_)
    {
        create(size_.height, size_.width, type_, data_, step_);
    }

    void copyTo(UMat& other) const
    {
        other.create(size(), type);
        cv::Mat src = toOpenCVMat();
        cv::Mat dst = other.toOpenCVMat();
        src.copyTo(dst);
    }

    UMat clone() const
    {
        UMat other;
        copyTo(other);
        return other;
    }

    cv::Mat toOpenCVMat() const
    {
        if (data)
            return cv::Mat(rows, cols, type, data, step);
        else
            return cv::Mat();
    }

    void upload(const cv::Mat& mat)
    {
        CV_Assert(mat.data);
        create(mat.size(), mat.type());
        cv::Mat header = toOpenCVMat();
        mat.copyTo(header);
    }

    void download(cv::Mat& mat) const
    {
        CV_Assert(data);
        mat.create(size(), type);
        cv::Mat header = toOpenCVMat();
        header.copyTo(mat);
    }

    void setZero()
    {
        if (data)
        {
            int actualLineSize = CV_MAT_CN(type) * (1 << (CV_MAT_DEPTH(type) / 2)) * cols;
            for (int i = 0; i < rows; i++)
                memset(data + i * step, 0, actualLineSize);
        }
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
        if (!data)
            return 0;
        return CV_MAT_CN(type);
    }

    unsigned char* data;
    int rows, cols;
    int step;
    int type;
    std::shared_ptr<unsigned char> sdata;

    cl_context ctx;
    cl_mem mem;
    std::shared_ptr<_cl_mem> smem;
};

}