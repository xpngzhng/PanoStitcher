#pragma once

#include "OpenCLAccel/basic.hpp"
#include "opencv2/core.hpp"
#include <memory>

#include "RunTimeObjects.h"

struct OpenCLQueue
{
public:
    OpenCLQueue(const OpenCLBasic& ocl)
    {
        CV_Assert(ocl.context);
        cl_int err = 0;
        queue = clCreateCommandQueue(ocl.context, ocl.device, 0, &err);
        SAMPLE_CHECK_ERRORS(err);
    }

    ~OpenCLQueue()
    {
        cl_int err = clReleaseCommandQueue(queue);
        SAMPLE_CHECK_ERRORS(err);
    }

    void waitForCompletion()
    {
        cl_int err = clFinish(queue);
        SAMPLE_CHECK_ERRORS(err);
    }

    cl_command_queue queue;

private:
    OpenCLQueue(const OpenCLQueue&);
    OpenCLQueue& operator=(const OpenCLQueue&);
};

namespace docl
{

struct HostMem
{
    enum BufferFlag
    {
        BufferFlagReadWrite = 1,
        BufferFlagWriteOnly = 2,
        BufferFlagReadOnly = 4,
        BufferFlagUseHostPtr = 8,
        BufferFlagAllocHostPtr = 16,
        BufferFlagCopyHostPtr = 32
    };

    enum MapFlag
    {
        MapFlagRead = 1,
        MapFlagWrite = 2
    };

    HostMem()
    {
        ctx = 0;
        mem = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
        data = 0;

        bufferFlag = -1;
        mapFlag = -1;
    }

    HostMem(int rows, int cols, int type, int bufferFlag, int mapFlag)
    {
        init();
        create(rows, cols, type, bufferFlag, mapFlag);
    }

    HostMem(const cv::Size& size, int type, int bufferFlag, int mapFlag)
    {
        init();
        create(size, type, bufferFlag, mapFlag);
    }

    // Notice, if we use locked as a member of HostMem and there is no destructor,
    // the program will crash when finishes in release mode, perhaps because undefined 
    // sequence of members' destructors calling.
    // We must make sure that mdata is destructed before smem is destructed.
    ~HostMem()
    {
        clear();
    }

    void init()
    {
        ctx = 0;
        mem = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;
        data = 0;

        bufferFlag = -1;
        mapFlag = -1;
    }

    void clear()
    {
        data = 0;
        mdata.reset();

        ctx = 0;
        mem = 0;
        smem.reset();

        rows = 0;
        cols = 0;
        step = 0;
        type = 0;

        bufferFlag = -1;
        mapFlag = -1;
    }

    void release()
    {
        clear();
    }

    void create(int rows_, int cols_, int type_, int bufferFlag_, int mapFlag_)
    {
        if (rows == rows_ && cols == cols_ && type == (type_& CV_MAT_TYPE_MASK) &&
            bufferFlag == bufferFlag_ && mapFlag == mapFlag_)
            return;

        clear();
        if (rows_ <= 0 || cols_ <= 0)
            return;

        rows = rows_;
        cols = cols_;
        type = type_ & CV_MAT_TYPE_MASK;
        CV_Assert(ocl && ocl->context);
        ctx = ocl->context;
        int channels = CV_MAT_CN(type);
        int elemSize1 = 1 << (CV_MAT_DEPTH(type) / 2);
        step = elemSize1 * channels * cols;

        int err = 0;

        bufferFlag = bufferFlag_;
        mem = clCreateBuffer(ctx, bufferFlag, rows * step, 0, &err);
        if (err) clear();
        SAMPLE_CHECK_ERRORS(err);
        smem.reset(mem, clReleaseMemObject);

        mapFlag = mapFlag_;
        data = (unsigned char*)clEnqueueMapBuffer(ocl->queue, mem, CL_TRUE, mapFlag, 0, step * rows, 0, 0, 0, &err);
        if (err) clear();
        SAMPLE_CHECK_ERRORS(err);
        mdata.reset(new MappedData(data, mem));
    }

    void create(const cv::Size& size_, int type_, int bufferFlag_, int mapFlag_)
    {
        create(size_.height, size_.width, type_, bufferFlag_, mapFlag_);
    }

    void copyTo(HostMem& other) const
    {
        other.create(size(), type, bufferFlag, mapFlag);
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

    void lock() const
    {
        if (data)
        {
            int err = 0;
            err = clEnqueueUnmapMemObject(ocl->queue, mem, data, 0, 0, 0);
            SAMPLE_CHECK_ERRORS(err);
            err = clFinish(ocl->queue);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    void unlock() const
    {
        if (data)
        {
            int err;
            void * ret = clEnqueueMapBuffer(ocl->queue, mem, CL_TRUE, mapFlag, 0, step * rows, 0, 0, 0, &err);
            SAMPLE_CHECK_ERRORS(err);
            CV_Assert(ret == data);
        }
    }

    void lock(OpenCLQueue& q) const
    {
        if (data)
        {
            int err = 0;
            err = clEnqueueUnmapMemObject(q.queue, mem, data, 0, 0, 0);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    void unlock(OpenCLQueue& q) const
    {
        if (data)
        {
            int err;
            void * ret = clEnqueueMapBuffer(q.queue, mem, CL_TRUE, mapFlag, 0, step * rows, 0, 0, 0, &err);
            SAMPLE_CHECK_ERRORS(err);
            CV_Assert(ret == data);
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
        if (!mem)
            return 0;
        return CV_MAT_CN(type);
    }

    unsigned char* data;
    struct MappedData
    {
        MappedData(void* data_, cl_mem mem_)
            : data(data_), mem(mem_) {}
        ~MappedData()
        {
            clEnqueueUnmapMemObject(ocl->queue, mem, data, 0, 0, 0);
        }

        void* data;
        cl_mem mem;
    };
    std::shared_ptr<MappedData> mdata;

    int rows, cols;
    int step;
    int type;

    cl_context ctx;
    cl_mem mem;
    std::shared_ptr<_cl_mem> smem;

    int bufferFlag;
    int mapFlag;
};

struct HostMemLockGuard
{
    HostMemLockGuard(const HostMem& hostMem_) : hostMem(hostMem_)
    {
        hostMem.lock();
    }
    ~HostMemLockGuard()
    {
        hostMem.unlock();
    }
    const HostMem& hostMem;
};

struct HostMemQueuedLockGuard
{
    HostMemQueuedLockGuard(const HostMem& hostMem_, OpenCLQueue& q_) : hostMem(hostMem_), q(q_)
    {
        hostMem.lock(q);
    }
    ~HostMemQueuedLockGuard()
    {
        hostMem.unlock(q);
    }
    const HostMem& hostMem;
    OpenCLQueue& q;
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
        init();
        create(rows, cols, type);
    }

    GpuMat(const cv::Size& size, int type)
    {
        init();
        create(size, type);
    }

    GpuMat(int rows, int cols, int type, cl_mem data, int step)
    {
        init();
        create(rows, cols, type, data, step);
    }

    GpuMat(const cv::Size& size, int type, cl_mem data, int step)
    {
        init();
        create(size.height, size.width, type, data, step);
    }

    void init()
    {
        mem = 0;
        data = 0;
        rows = 0;
        cols = 0;
        step = 0;
        type = 0;

        ctx = 0;
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

        clear();
        if (rows_ <= 0 || cols_ <= 0)
            return;

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
        {
            HostMemLockGuard lg(mat);
            int err = 0;
            err = clEnqueueCopyBuffer(ocl->queue, mat.mem, mem, 0, 0, step * rows, 0, 0, 0);
            SAMPLE_CHECK_ERRORS(err);
            err = clFinish(ocl->queue);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    void download(HostMem& mat, int bufferFlag, int mapFlag) const
    {
        CV_Assert(mem);
        mat.create(rows, cols, type, bufferFlag, mapFlag);
        {
            HostMemLockGuard lg(mat);
            int err = 0;
            err = clEnqueueCopyBuffer(ocl->queue, mem, mat.mem, 0, 0, step * rows, 0, 0, 0);
            SAMPLE_CHECK_ERRORS(err);
            err = clFinish(ocl->queue);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    void upload(const HostMem& mat, OpenCLQueue& q)
    {
        CV_Assert(mat.mem);
        create(mat.rows, mat.cols, mat.type);
        {
            HostMemQueuedLockGuard lg(mat, q);
            int err = 0;
            err = clEnqueueCopyBuffer(q.queue, mat.mem, mem, 0, 0, step * rows, 0, 0, 0);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    void download(HostMem& mat, int bufferFlag, int mapFlag, OpenCLQueue& q) const
    {
        CV_Assert(mem);
        mat.create(rows, cols, type, bufferFlag, mapFlag);
        {
            HostMemQueuedLockGuard lg(mat, q);
            int err = 0;
            err = clEnqueueCopyBuffer(q.queue, mem, mat.mem, 0, 0, step * rows, 0, 0, 0);
            SAMPLE_CHECK_ERRORS(err);
        }
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

    void upload(const cv::Mat& mat, OpenCLQueue& q)
    {
        CV_Assert(mat.data);

        create(mat.rows, mat.cols, mat.type());
        int err = 0;
        if (mat.isContinuous())
            err = clEnqueueWriteBuffer(q.queue, mem, CL_FALSE, 0, step * rows, mat.data, 0, 0, 0);
        else
        {
            size_t deviceOrigin[3] = { 0, 0, 0 };
            size_t hostOrigin[3] = { 0, 0, 0 };
            size_t region[3] = { step, rows, 1 };
            err = clEnqueueWriteBufferRect(q.queue, mem, CL_FALSE, deviceOrigin, hostOrigin, region,
                step, 1, mat.step, 1, mat.data, 0, 0, 0);
        }
        SAMPLE_CHECK_ERRORS(err);
    }

    void download(cv::Mat& mat, OpenCLQueue& q) const
    {
        CV_Assert(mem);

        mat.create(rows, cols, type);
        int err = 0;
        if (mat.isContinuous())
            err = clEnqueueReadBuffer(q.queue, mem, CL_FALSE, 0, step * rows, mat.data, 0, 0, 0);
        else
        {
            size_t deviceOrigin[3] = { 0, 0, 0 };
            size_t hostOrigin[3] = { 0, 0, 0 };
            size_t region[3] = { step, rows, 1 };
            err = clEnqueueReadBufferRect(q.queue, mem, CL_FALSE, deviceOrigin, hostOrigin, region,
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