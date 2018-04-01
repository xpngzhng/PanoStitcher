# PanoSticher 
全景图和全景视频拼接项目，和 EasyFFmpeg 搭配使用

## 主要内容
1. [Blend](https://github.com/XupingZHENG/PanoSticher/tree/master/source/Blend) 图像融合模块  
1.1 融合算法 -- 线性融合 Linear Blend，多分辨率融合 Multiband Blend，多分辨率融合算法有多线程实现版本  
1.2 缝合线查找算法 -- 基于距离变换 Distance Transform 的算法，基于图割 Graph Cut 的算法，基于图割的算法有多尺度实现版本  
1.3 曝光与白平衡校正算法
2. [Warp](https://github.com/XupingZHENG/PanoSticher/tree/master/source/Warp) 图像变形模块  
2.1 重投影算法 -- RectLinear，EuquiRectangular，FishEye，Little Planet 格式的相互转换  
2.2 全景视频去抖动算法  
3. [CudaAccel](https://github.com/XupingZHENG/PanoSticher/tree/master/source/CudaAccel) CUDA 加速模块  
使用 CUDA 实现图像融合和图像变形的算法  
4. [OpenCLAccel](https://github.com/XupingZHENG/PanoSticher/tree/master/source/OpenCLAccel) OpenCL 加速基础模块
对 OpenCL 运行时库进行封装
5. [DiscreteOpenCL](https://github.com/XupingZHENG/PanoSticher/tree/master/source/DiscreteOpenCL) 独立显卡 OpenCL 加速模块  
使用 OpenCL 实现图像融合和图像变形算法，用于 NVIDIA 和 AMD 的独立显卡  
6. [IntelOpenCL](https://github.com/XupingZHENG/PanoSticher/tree/master/source/IntelOpenCL) Intel 集成显卡 OpenCL 加速模块  
使用 OpenCL 实现图像融合和图像变形算法，用于 Intel 的集成显卡  
7. [Task](https://github.com/XupingZHENG/PanoSticher/tree/master/source/Task) 任务模块  
7.1 全景视频拼接任务 -- 离线制作全景视频并保存到本地  
7.2 全景视频直播任务 -- 实时制作全景视频并进行推流
8. [Tool](https://github.com/XupingZHENG/PanoSticher/tree/master/source/Tool) 工具模块  
辅助工具
