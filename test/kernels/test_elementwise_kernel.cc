#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace infini {

// 线程测试参数 - 模板类支持任意数据类型
template <typename T> struct ThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    OpType opType = OpType::Unknown;
    Shape shapeA;
    Shape shapeB;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputAData;
    std::vector<T> inputBData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// 设备线程函数 - 模板函数
template <typename T> void deviceThreadFunc(ThreadTestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // 初始化设备 Context
    runtime->initThreadContext(params.device, params.deviceId);

    // 创建 Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(params.shapeA, params.dataType);
    auto B = g->addTensor(params.shapeB, params.dataType);
    auto op = g->addOp<ElementWiseObj>(params.opType, A, B, nullptr);

    // 先设置数据（设置CPU指针），再分配内存（触发H2D拷贝）
    A->setData(params.inputAData.data());
    B->setData(params.inputBData.data());
    runtime->dataMalloc(g); // 会检测到data是CPU，执行H2D拷贝

    // 运行计算
    runtime->run(g);

    // 获取输出并复制到 host
    auto output = op->getOutput(0);
    size_t numElements = output->getElement();
    params.outputData.resize(numElements);

    // 检查 output 的数据是否存在
    auto dataBlob = output->getData();
    if (!dataBlob) {
        throw std::runtime_error("Output data blob is null!");
    }
    void *devicePtr = dataBlob->getRawDataPtr();
    if (!devicePtr && !runtime->isCpu()) {
        throw std::runtime_error(
            "Output device pointer is null on GPU device!");
    }

    // 复制结果数据
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);

    // 根据数据类型复制
    if constexpr (std::is_same_v<T, float>) {
        if (params.dataType.getType() == INFINI_DTYPE_F32) {
            std::memcpy(params.outputData.data(), hostPtr,
                        numElements * sizeof(float));
        } else if (params.dataType.getType() == INFINI_DTYPE_F16) {
            // FP16 转换为 FP32
            uint16_t *fp16Data = static_cast<uint16_t *>(hostPtr);
            for (size_t i = 0; i < numElements; ++i) {
                params.outputData[i] = fp16_to_fp32(fp16Data[i]);
            }
        }
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        if (params.dataType.getType() == INFINI_DTYPE_F16) {
            std::memcpy(params.outputData.data(), hostPtr,
                        numElements * sizeof(uint16_t));
        } else if (params.dataType.getType() == INFINI_DTYPE_F32) {
            // FP32 转换为 FP16
            float *fp32Data = static_cast<float *>(hostPtr);
            uint16_t *fp16Data = params.outputData.data();
            for (size_t i = 0; i < numElements; ++i) {
                // 这里需要 FP32 转 FP16 的函数，暂时用简单的方式
                fp16Data[i] = static_cast<uint16_t>(fp32Data[i]);
            }
        }
    }

    runtime->deallocHost(hostPtr);
    params.completed = true;
}

// 运行多线程测试 - 模板函数
template <typename T>
void runMultiThreadTest(OpType opType, const Shape &shapeA, const Shape &shapeB,
                        const DataType &dataType, bool print = false) {

    // 准备输入数据
    size_t elementA = 1, elementB = 1;
    for (auto dim : shapeA)
        elementA *= dim;
    for (auto dim : shapeB)
        elementB *= dim;

    std::vector<T> inputAData(elementA);
    std::vector<T> inputBData(elementB);

    // 使用简单的递增序列和递减序列，便于计算和验证
    for (size_t i = 0; i < elementA; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            inputAData[i] = static_cast<T>(i + 1); // 1, 2, 3, ...
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            inputAData[i] = static_cast<T>(i + 1);
        }
    }
    for (size_t i = 0; i < elementB; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            inputBData[i] = static_cast<T>(elementB - i); // n, n-1, n-2, ...
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            inputBData[i] = static_cast<T>(elementB - i);
        }
    }

    // 创建线程参数
    ThreadTestParams<T> cpuParams, gpuParams;

    // CPU 线程参数
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.opType = opType;
    cpuParams.shapeA = shapeA;
    cpuParams.shapeB = shapeB;
    cpuParams.dataType = dataType;
    cpuParams.inputAData = inputAData;
    cpuParams.inputBData = inputBData;
    cpuParams.deviceName = "CPU";

    // GPU 线程参数
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 0;
    gpuParams.opType = opType;
    gpuParams.shapeA = shapeA;
    gpuParams.shapeB = shapeB;
    gpuParams.dataType = dataType;
    gpuParams.inputAData = inputAData;
    gpuParams.inputBData = inputBData;
    gpuParams.deviceName = "NVIDIA";

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread ElementWise Test" << std::endl;
        std::cout << "OpType: " << opType.toString() << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape A: " << vecToString(shapeA) << std::endl;
        std::cout << "Shape B: " << vecToString(shapeB) << std::endl;
        std::cout << "Thread 1: CPU (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "Thread 2: NVIDIA (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // 启动两个线程并行执行
    std::thread cpuThread(deviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(deviceThreadFunc<T>, std::ref(gpuParams));

    // 等待两个线程完成
    cpuThread.join();
    gpuThread.join();

    // 验证结果
    ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
    ASSERT_TRUE(gpuParams.completed) << "NVIDIA thread failed";

    ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size())
        << "Output size mismatch";

    // 对比结果
    size_t numErrors = 0;
    float maxError = 0.0f;
    const float epsilon = 1e-3f;

    for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
        float cpuVal, gpuVal;

        // 转换为 float 进行比较
        if constexpr (std::is_same_v<T, float>) {
            cpuVal = cpuParams.outputData[i];
            gpuVal = gpuParams.outputData[i];
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            // FP16 转 FP32 比较
            cpuVal = fp16_to_fp32(cpuParams.outputData[i]);
            gpuVal = fp16_to_fp32(gpuParams.outputData[i]);
        }

        float error = std::abs(cpuVal - gpuVal);
        maxError = std::max(maxError, error);

        if (error > epsilon) {
            numErrors++;
            if (numErrors <= 5) { // 只打印前5个错误
                std::cout << "Mismatch at index " << i << ": CPU=" << cpuVal
                          << ", NVIDIA=" << gpuVal << ", error=" << error
                          << std::endl;
            }
        }
    }

    if (print) {
        std::cout << "Result Comparison:" << std::endl;
        std::cout << "  Total elements: " << cpuParams.outputData.size()
                  << std::endl;
        std::cout << "  Errors: " << numErrors << std::endl;
        std::cout << "  Max error: " << maxError << std::endl;

        if (numErrors == 0) {
            std::cout << "  ✓ Test PASSED" << std::endl;
        } else {
            std::cout << "  ✗ Test FAILED" << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0)
        << "Results mismatch between CPU and NVIDIA (max error: " << maxError
        << ")";
}

// 基本Add操作测试 - F32
TEST(ElementWise, Add_MultiThread_F32) {
    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<float>(OpType::Add, shapeA, shapeB,
                              DataType(INFINI_DTYPE_F32), true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// 基本Add操作测试 - F16
TEST(ElementWise, Add_MultiThread_F16) {
    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(OpType::Add, shapeA, shapeB,
                                 DataType(INFINI_DTYPE_F16), true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// 基本Mul操作测试 - F32
TEST(ElementWise, Mul_MultiThread_F32) {
    Shape shapeA = {3, 4};
    Shape shapeB = {3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<float>(OpType::Mul, shapeA, shapeB,
                              DataType(INFINI_DTYPE_F32), false);
#endif
}

// 基本Mul操作测试 - F16
TEST(ElementWise, Mul_MultiThread_F16) {
    Shape shapeA = {3, 4};
    Shape shapeB = {3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(OpType::Mul, shapeA, shapeB,
                                 DataType(INFINI_DTYPE_F16), false);
#endif
}

// 基本Sub操作测试 - F32
TEST(ElementWise, Sub_MultiThread_F32) {
    Shape shapeA = {1, 5, 6};
    Shape shapeB = {1, 5, 6};

#ifdef USE_CUDA
    runMultiThreadTest<float>(OpType::Sub, shapeA, shapeB,
                              DataType(INFINI_DTYPE_F32), false);
#endif
}

// 基本Sub操作测试 - F16
TEST(ElementWise, Sub_MultiThread_F16) {
    Shape shapeA = {1, 5, 6};
    Shape shapeB = {1, 5, 6};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(OpType::Sub, shapeA, shapeB,
                                 DataType(INFINI_DTYPE_F16), false);
#endif
}

// 单设备测试（用于调试）- CPU
TEST(ElementWise, Add_SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = g->addTensor(shapeB, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    runtime->dataMalloc(g);

    // 设置输入数据
    std::vector<float> inputAData(A->getElement());
    std::vector<float> inputBData(B->getElement());

    for (size_t i = 0; i < inputAData.size(); ++i) {
        inputAData[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < inputBData.size(); ++i) {
        inputBData[i] = static_cast<float>(inputBData.size() - i);
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());

    // 执行计算
    runtime->run(g);

    // 获取输出并打印
    auto output = op->getOutput(0);
    std::cout << "CPU Output Data: " << std::endl;
    output->printData(runtime);
}

#ifdef USE_CUDA
// 单设备测试（用于调试）- NVIDIA F32
TEST(ElementWise, Add_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = g->addTensor(shapeB, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    // 设置输入数据
    std::vector<float> inputAData(A->getElement());
    std::vector<float> inputBData(B->getElement());

    for (size_t i = 0; i < inputAData.size(); ++i) {
        inputAData[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < inputBData.size(); ++i) {
        inputBData[i] = static_cast<float>(inputBData.size() - i);
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());
    runtime->dataMalloc(g);

    // 执行计算
    runtime->run(g);

    // 获取输出并打印
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Output Data: " << std::endl;
    output->printData(runtime);
}

// 单设备测试（用于调试）- NVIDIA F16
TEST(ElementWise, Add_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType(INFINI_DTYPE_F16));
    auto B = g->addTensor(shapeB, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    // 设置输入数据
    std::vector<uint16_t> inputAData(A->getElement());
    std::vector<uint16_t> inputBData(B->getElement());

    for (size_t i = 0; i < inputAData.size(); ++i) {
        inputAData[i] = static_cast<uint16_t>(i + 1);
    }
    for (size_t i = 0; i < inputBData.size(); ++i) {
        inputBData[i] = static_cast<uint16_t>(inputBData.size() - i);
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());
    runtime->dataMalloc(g);

    // 执行计算
    runtime->run(g);

    // 获取输出并打印
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 Output Data: " << std::endl;
    output->printData(runtime);
}
#endif

} // namespace infini
