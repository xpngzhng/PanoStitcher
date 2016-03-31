#include "CudaUtil.cuh"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define PYR_DOWN_BLOCK_WIDTH 16
#define PYR_DOWN_BLOCK_HEIGHT 16
#define PYR_UP_BLOCK_WIDTH 16
#define PYR_UP_BLOCK_HEIGHT 16
#define UTIL_BLOCK_WIDTH 32
#define UTIL_BLOCK_HEIGHT 8

// blockWidth counted by auxWidth, blockHeight counted by srcHeight
__global__ void pyrHoriDown16SC1To32SC1(const unsigned char* srcData, int srcRows, int srcCols, int srcStep,
    unsigned char* auxData, int auxRows, int auxCols, int auxStep, const int* horiIndex)
{
    __shared__ int idxBuf[PYR_DOWN_BLOCK_WIDTH * 4];

    int auxx = blockDim.x * blockIdx.x + threadIdx.x;
    int srcx = auxx * 2;
    int srcy = blockDim.y * blockIdx.y + threadIdx.y;
    int threadx = threadIdx.x;
    int ofsx = blockDim.x * blockIdx.x * 2 + threadx;
    idxBuf[threadx                           ] = horiIndex[ofsx                           ];
    idxBuf[threadx + PYR_DOWN_BLOCK_WIDTH    ] = horiIndex[ofsx + PYR_DOWN_BLOCK_WIDTH    ];
    idxBuf[threadx + PYR_DOWN_BLOCK_WIDTH * 2] = horiIndex[ofsx + PYR_DOWN_BLOCK_WIDTH * 2];
    idxBuf[threadx + PYR_DOWN_BLOCK_WIDTH * 3] = horiIndex[ofsx + PYR_DOWN_BLOCK_WIDTH * 3];
    __syncthreads();
    
    int idx = threadx * 2 + PYR_DOWN_BLOCK_WIDTH - 2;
    const short* ptr = getRowPtr<short>(srcData, srcStep, srcy);
    int sum = 0;
    sum  =     ptr[idxBuf[idx++]];
    sum += 4 * ptr[idxBuf[idx++]];
    sum += 6 * ptr[idxBuf[idx++]];
    sum += 4 * ptr[idxBuf[idx++]];
    sum +=     ptr[idxBuf[idx  ]];

    if (srcy < auxRows && auxx < auxCols)
        getRowPtr<int>(auxData, auxStep, srcy)[auxx] = sum;
}

// blockWidth counted by auxWidth, blockHeight counted by dstHeight
__global__ void pyrVertDown32SC1To32SC1(const unsigned char* auxData, int auxRows, int auxCols, int auxStep,
    unsigned char* dstData, int dstRows, int dstCols, int dstStep, const int* vertIndex)
{
    __shared__ int idxBuf[PYR_DOWN_BLOCK_HEIGHT * 4];

    int auxx = blockDim.x * blockIdx.x + threadIdx.x;
    int dsty = blockDim.y * blockIdx.y + threadIdx.y;
    int auxy = dsty * 2;
    int thready = threadIdx.y;
    int ofsy = blockDim.y * blockIdx.y * 2 + thready;
    idxBuf[thready                            ] = vertIndex[ofsy                            ];
    idxBuf[thready + PYR_DOWN_BLOCK_HEIGHT    ] = vertIndex[ofsy + PYR_DOWN_BLOCK_HEIGHT    ];
    idxBuf[thready + PYR_DOWN_BLOCK_HEIGHT * 2] = vertIndex[ofsy + PYR_DOWN_BLOCK_HEIGHT * 2];
    idxBuf[thready + PYR_DOWN_BLOCK_HEIGHT * 3] = vertIndex[ofsy + PYR_DOWN_BLOCK_HEIGHT * 3];
    __syncthreads();

    int idx = thready * 2 + PYR_DOWN_BLOCK_HEIGHT - 2;
    int sum = 0;
    sum  =     getElem<int>(auxData, auxStep, idxBuf[idx++], auxx);
    sum += 4 * getElem<int>(auxData, auxStep, idxBuf[idx++], auxx);
    sum += 6 * getElem<int>(auxData, auxStep, idxBuf[idx++], auxx);
    sum += 4 * getElem<int>(auxData, auxStep, idxBuf[idx++], auxx);
    sum +=     getElem<int>(auxData, auxStep, idxBuf[idx  ], auxx);

    if (dsty < dstRows && auxx < dstCols)
        getRowPtr<int>(dstData, dstStep, dsty)[auxx] = sum;
}

void pyramidDown16SC1To32SC1(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& aux,
    const cv::cuda::GpuMat& horiIndexTab, const cv::cuda::GpuMat& vertIndexTab)
{
    CV_Assert(src.data && src.type() == CV_16SC1 &&
        dst.data && dst.type() == CV_32SC1 &&
        aux.data && aux.type() == CV_32SC1 &&
        horiIndexTab.data && horiIndexTab.type() == CV_32SC1 &&
        vertIndexTab.data && vertIndexTab.type() == CV_32SC1);

    int srcRows = src.rows, srcCols = src.cols;
    int dstRows = (srcRows + 1) >> 1, dstCols = (srcCols + 1) >> 1;
    CV_Assert(dst.rows == dstRows && dst.cols == dstCols &&
        aux.rows == srcRows && aux.cols == dstCols &&
        horiIndexTab.cols == cv::cuda::device::divUp(srcCols, PYR_DOWN_BLOCK_WIDTH) * PYR_DOWN_BLOCK_WIDTH + 2 * PYR_DOWN_BLOCK_WIDTH &&
        vertIndexTab.cols == cv::cuda::device::divUp(srcRows, PYR_DOWN_BLOCK_HEIGHT) * PYR_DOWN_BLOCK_HEIGHT + 2 * PYR_DOWN_BLOCK_HEIGHT);

    dim3 block(PYR_DOWN_BLOCK_WIDTH, PYR_DOWN_BLOCK_HEIGHT);
    dim3 grid(cv::cuda::device::divUp(dst.cols, block.x), cv::cuda::device::divUp(src.rows, block.y));
    pyrHoriDown16SC1To32SC1<<<grid, block>>>(src.data, src.rows, src.cols, src.step, 
        aux.data, aux.rows, aux.cols, aux.step, (int*)horiIndexTab.data);
    grid.y = cv::cuda::device::divUp(dst.rows, block.y);
    pyrVertDown32SC1To32SC1<<<grid, block>>>(aux.data, aux.rows, aux.cols, aux.step,
        dst.data, dst.rows, dst.cols, dst.step, (int*)vertIndexTab.data);
    //cudaSafeCall(cudaGetLastError());
    //cudaSafeCall(cudaDeviceSynchronize());
}

