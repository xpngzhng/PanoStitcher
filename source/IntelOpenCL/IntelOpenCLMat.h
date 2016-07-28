#pragma once

#include "opencv2/core.hpp"
#include "CL/cl.h"
#include <memory>

const int ptrAlignSize = 128;
const int stepAlignSize = 128;

struct IntelOclMat
{
    IntelOclMat()
    {
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
    }

    IntelOclMat(int rows, int cols, int type)
    {
        create(rows, cols, type);
    }

    void clear()
    {
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
        sdata.reset();
    }

    void create(int rows_, int cols_, int type_)
    {
        if (rows_ <= 0 || cols_ <= 0)
        {
            clear();
            return;
        }

        if (rows != rows_ || cols != cols_ || type != (type_& CV_MAT_TYPE_MASK))
        {
            sdata.reset();
            rows = rows_;
            cols = cols_;
            type = type_ & CV_MAT_TYPE_MASK;
            int channels = CV_MAT_CN(type);
            int elemSize1 = 1 << (CV_MAT_DEPTH(type) / 2);
            step = (elemSize1 * channels * cols + stepAlignSize - 1) / stepAlignSize * stepAlignSize;
            data = (unsigned char*)_aligned_malloc(step * rows, ptrAlignSize);
            sdata.reset(data, _aligned_free);
            return;
        }
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

    cv::Mat toOpenCVMat() const
    {
        if (data)
            return cv::Mat(rows, cols, type, data, step);
        else
            return cv::Mat();
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
};

struct IOclMat
{
    IOclMat()
    {
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;

        ctx = 0;
        mem = 0;
    }

    IOclMat(int rows, int cols, int type, cl_context ctx)
    {
        create(rows, cols, type, ctx);
    }

    IOclMat(const cv::Size& size, int type, cl_context ctx)
    {
        create(size, type, ctx);
    }

    IOclMat(int rows, int cols, int type, unsigned char* data, int step, cl_context ctx)
    {
        create(rows, cols, type, data, step, ctx);
    }

    IOclMat(const cv::Size& size, int type, unsigned char* data, int step, cl_context ctx)
    {
        create(size, type, data, step, ctx);
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

    void create(int rows_, int cols_, int type_, cl_context ctx_)
    {
        if (rows == rows_ && cols == cols_ && type == (type_& CV_MAT_TYPE_MASK) && ctx == ctx_)
            return;

        if (rows_ <= 0 || cols_ <= 0 || !ctx_)
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

        if (dataChanged || ctx != ctx_)
        {
            smem.reset();
            ctx = ctx_;
            int err = 0;
            mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * step, data, &err);
            smem.reset(mem, clReleaseMemObject);
            if (err)
            {
                clear();
            }
        }
    }

    void create(const cv::Size& size_, int type_, cl_context ctx_)
    {
        create(size_.height, size_.width, type_, ctx_);
    }

    void create(int rows_, int cols_, int type_, unsigned char* data_, int step_, cl_context ctx_)
    {
        clear();

        if (rows_ <= 0 || cols_ <= 0 || !ctx_)
            return;

        rows = rows_;
        cols = cols_;
        step = step_;
        type = type_ & CV_MAT_TYPE_MASK;
        data = data_;
        ctx = ctx_;
        int err = 0;
        mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * step, data, &err);
        smem.reset(mem, clReleaseMemObject);
        if (err)
            clear();
    }

    void create(const cv::Size& size_, int type_, unsigned char* data_, int step_, cl_context ctx_)
    {
        create(size_.height, size_.width, type_, data_, step_, ctx_);
    }

    void copyTo(IOclMat& other) const
    {
        other.create(size(), type, ctx);
        cv::Mat src = toOpenCVMat();
        cv::Mat dst = other.toOpenCVMat();
        src.copyTo(dst);
    }

    IOclMat clone() const
    {
        IOclMat other;
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

    void upload(const cv::Mat& mat, cl_context ctx_)
    {
        CV_Assert(mat.data);
        create(mat.size(), mat.type(), ctx_);
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