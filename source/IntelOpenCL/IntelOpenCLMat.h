#pragma once

#include "opencv2/core.hpp"
#include "CL/cl.h"
#include <memory>

const int ptrAlignSize = 4096;
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
        image = 0;
        sampler = 0;
    }

    IOclMat(int rows, int cols, int type, cl_context ctx)
    {
        create(rows, cols, type, ctx);
    }

    IOclMat(const cv::Size& size, int type, cl_context ctx)
    {
        create(size, type, ctx);
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
        image = 0; 
        simage.reset();
        sampler = 0;
        ssampler = 0;
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

        // If memory reallocated, or context changed, image and sampler are invalidated
        image = 0;
        simage.reset();
        sampler = 0;
        ssampler.reset();
    }

    void create(const cv::Size& size_, int type_, cl_context ctx_)
    {
        create(size_.height, size_.width, type_, ctx_);
    }

    bool bindReadOnlyImageAndSampler()
    {
        simage.reset();
        ssampler.reset();
        image = 0;
        sampler = 0;
        if (type == CV_8UC4)
        {
            int err = 0;

            cl_image_format clImageFormat;
            clImageFormat.image_channel_order = CL_RGBA;
            clImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
            image = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &clImageFormat,
                cols, rows, step, data, &err);
            if (err)
            {
                image = 0;
                return false;
            }                
            simage.reset(image, clReleaseMemObject);

            sampler = clCreateSampler(ctx, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
            if (err)
            {
                image = 0;
                simage.reset();
                sampler = 0;
                return false;
            }
            ssampler.reset(sampler, clReleaseSampler);

            return true;
        }
        return false;
    }

    cv::Mat toOpenCVMat() const
    {
        if (data)
            return cv::Mat(rows, cols, type, data, step);
        else
            return cv::Mat();
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
    cl_mem image;
    std::shared_ptr<_cl_mem> simage;
    cl_sampler sampler;
    std::shared_ptr<_cl_sampler> ssampler;
};