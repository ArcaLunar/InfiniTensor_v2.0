#include "core/runtime.h"
#include "operators/Clip.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <type_traits>

namespace infini {

template <typename T> struct ClipThreadTestParams {
	infiniDevice_t device = INFINI_DEVICE_CPU;
	int deviceId = 0;
	Shape inputShape;
	DataType dataType = DataType(INFINI_DTYPE_F32);
	std::vector<T> inputData;
	std::vector<T> minData;
	std::vector<T> maxData;
	std::vector<T> outputData;
	bool completed = false;
};

template <typename T>
float toFloatValue(T val) {
	if constexpr (std::is_same_v<T, float>) {
		return val;
	} else if constexpr (std::is_same_v<T, uint16_t>) {
		return fp16_to_fp32(val);
	} else {
		static_assert(std::is_same_v<T, float> ||
						  std::is_same_v<T, uint16_t>,
					  "Unsupported type for toFloatValue");
		return 0.0f;
	}
}

template <typename T>
void clipDeviceThreadFunc(ClipThreadTestParams<T> &params) {
	RuntimeObj::init();
	Runtime &runtime = RuntimeObj::getInstance();

	runtime->initThreadContext(params.device, params.deviceId);

	Graph g = make_ref<GraphObj>(runtime);
	auto input = g->addTensor(params.inputShape, params.dataType);
	auto minTensor = g->addTensor(params.inputShape, params.dataType);
	auto maxTensor = g->addTensor(params.inputShape, params.dataType);
	auto op = g->addOp<ClipObj>(input, minTensor, maxTensor, nullptr);

	input->setData(params.inputData.data());
	minTensor->setData(params.minData.data());
	maxTensor->setData(params.maxData.data());
	runtime->dataMalloc(g);
	runtime->run(g);

	auto output = op->getOutput(0);
	size_t numElements = output->getElement();
	params.outputData.resize(numElements);

	auto dataBlob = output->getData();
	if (!dataBlob) {
		throw std::runtime_error("Output data blob is null!");
	}
	void *devicePtr = dataBlob->getRawDataPtr();
	if (!devicePtr && !runtime->isCpu()) {
		throw std::runtime_error(
			"Output device pointer is null on GPU device!");
	}

	void *hostPtr = runtime->allocHost(output->getTotalBytes());
	runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
					INFINIRT_MEMCPY_D2H);
	copyAndConvertData(params.outputData, hostPtr, numElements,
					   params.dataType);
	runtime->deallocHost(hostPtr);

	params.completed = true;
}

template <typename T>
void runClipMultiThreadTest(const Shape &inputShape,
							const std::vector<T> &inputData,
							const std::vector<T> &minData,
							const std::vector<T> &maxData,
							const DataType &dataType,
							float epsilon = 1e-3f) {
	ClipThreadTestParams<T> cpuParams, gpuParams;
	ASSERT_EQ(inputData.size(), minData.size());
	ASSERT_EQ(inputData.size(), maxData.size());

	cpuParams.device = INFINI_DEVICE_CPU;
	cpuParams.deviceId = 0;
	cpuParams.inputShape = inputShape;
	cpuParams.dataType = dataType;
	cpuParams.inputData = inputData;
	cpuParams.minData = minData;
	cpuParams.maxData = maxData;

	gpuParams.device = INFINI_DEVICE_NVIDIA;
	gpuParams.deviceId = 0;
	gpuParams.inputShape = inputShape;
	gpuParams.dataType = dataType;
	gpuParams.inputData = inputData;
	gpuParams.minData = minData;
	gpuParams.maxData = maxData;

	std::thread cpuThread(clipDeviceThreadFunc<T>, std::ref(cpuParams));
	std::thread gpuThread(clipDeviceThreadFunc<T>, std::ref(gpuParams));

	cpuThread.join();
	gpuThread.join();

	ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
	ASSERT_TRUE(gpuParams.completed) << "NVIDIA thread failed";
	ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size())
		<< "Output size mismatch";

	size_t numErrors = 0;
	for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
		const float cpuVal = toFloatValue(cpuParams.outputData[i]);
		const float gpuVal = toFloatValue(gpuParams.outputData[i]);
		const float inputVal = toFloatValue(inputData[i]);
		const float minFloat = toFloatValue(minData[i]);
		const float maxFloat = toFloatValue(maxData[i]);
		const float refVal = std::min(std::max(inputVal, minFloat), maxFloat);

		if (std::abs(cpuVal - gpuVal) > epsilon ||
			std::abs(cpuVal - refVal) > epsilon ||
			std::abs(gpuVal - refVal) > epsilon) {
			++numErrors;
			if (numErrors <= 5) {
				std::cout << "Mismatch at index " << i << ": input="
						  << inputVal << ", ref=" << refVal
						  << ", cpu=" << cpuVal << ", gpu=" << gpuVal
						  << std::endl;
			}
		}
	}

	EXPECT_EQ(numErrors, 0) << "Clip results mismatch";
}

template <typename T>
void runClipSingleDeviceTest(infiniDevice_t device, const Shape &inputShape,
							 const std::vector<T> &inputData,
							 const std::vector<T> &minData,
							 const std::vector<T> &maxData,
							 const DataType &dataType,
							 float epsilon = 1e-3f) {
	RuntimeObj::init();
	Runtime &runtime = RuntimeObj::getInstance();
	runtime->initThreadContext(device, 0);
	ASSERT_EQ(inputData.size(), minData.size());
	ASSERT_EQ(inputData.size(), maxData.size());

	Graph g = make_ref<GraphObj>(runtime);
	auto input = g->addTensor(inputShape, dataType);
	auto minTensor = g->addTensor(inputShape, dataType);
	auto maxTensor = g->addTensor(inputShape, dataType);
	auto op = g->addOp<ClipObj>(input, minTensor, maxTensor, nullptr);

	input->setData(const_cast<T *>(inputData.data()));
	minTensor->setData(const_cast<T *>(minData.data()));
	maxTensor->setData(const_cast<T *>(maxData.data()));
	runtime->dataMalloc(g);
	runtime->run(g);

	auto output = op->getOutput(0);
	std::vector<T> outputData(output->getElement());
	void *hostPtr = runtime->allocHost(output->getTotalBytes());
	runtime->memcpy(hostPtr, output->getRawDataPtr<void *>(),
					output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
	copyAndConvertData(outputData, hostPtr, output->getElement(), dataType);
	runtime->deallocHost(hostPtr);

	for (size_t i = 0; i < outputData.size(); ++i) {
		const float inputVal = toFloatValue(inputData[i]);
		const float minFloat = toFloatValue(minData[i]);
		const float maxFloat = toFloatValue(maxData[i]);
		const float outputVal = toFloatValue(outputData[i]);
		const float refVal = std::min(std::max(inputVal, minFloat), maxFloat);
		EXPECT_NEAR(outputVal, refVal, epsilon);
	}
}

TEST(Clip, Basic_MultiThread_F32) {
	Shape inputShape = {2, 5};
	std::vector<float> inputData = {-3.0f, -1.0f, 0.0f, 0.8f, 2.5f,
									3.2f,  -4.0f, 1.1f, 5.0f, -0.4f};
	std::vector<float> minData(inputData.size(), -1.0f);
	std::vector<float> maxData(inputData.size(), 1.0f);

#ifdef USE_CUDA
	runClipMultiThreadTest<float>(inputShape, inputData, minData, maxData,
								  DataType(INFINI_DTYPE_F32));
#else
	std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

TEST(Clip, Basic_MultiThread_F16) {
	Shape inputShape = {2, 5};
	std::vector<float> inputFloat = {-2.0f, -1.2f, -0.5f, 0.4f, 1.6f,
									 2.8f,  -3.0f, 0.0f,  0.9f, -0.1f};
	std::vector<uint16_t> inputData(inputFloat.size());
	std::vector<uint16_t> minData(inputFloat.size(), fp32_to_fp16(-1.0f));
	std::vector<uint16_t> maxData(inputFloat.size(), fp32_to_fp16(1.0f));
	std::transform(inputFloat.begin(), inputFloat.end(), inputData.begin(),
				   [](float v) { return fp32_to_fp16(v); });

#ifdef USE_CUDA
	runClipMultiThreadTest<uint16_t>(inputShape, inputData, minData, maxData,
									 DataType(INFINI_DTYPE_F16), 1e-2f);
#else
	std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

TEST(Clip, SingleDevice_CPU_F32) {
	Shape inputShape = {2, 5};
	std::vector<float> inputData = {-3.0f, -1.0f, 0.0f, 0.8f, 2.5f,
									3.2f,  -4.0f, 1.1f, 5.0f, -0.4f};
	std::vector<float> minData(inputData.size(), -1.0f);
	std::vector<float> maxData(inputData.size(), 1.0f);
	runClipSingleDeviceTest<float>(INFINI_DEVICE_CPU, inputShape, inputData,
								   minData, maxData,
								   DataType(INFINI_DTYPE_F32));
}

#ifdef USE_CUDA
TEST(Clip, SingleDevice_NVIDIA_F32) {
	Shape inputShape = {2, 5};
	std::vector<float> inputData = {-3.0f, -1.0f, 0.0f, 0.8f, 2.5f,
									3.2f,  -4.0f, 1.1f, 5.0f, -0.4f};
	std::vector<float> minData(inputData.size(), -1.0f);
	std::vector<float> maxData(inputData.size(), 1.0f);
	runClipSingleDeviceTest<float>(INFINI_DEVICE_NVIDIA, inputShape, inputData,
								   minData, maxData,
								   DataType(INFINI_DTYPE_F32));
}

TEST(Clip, SingleDevice_NVIDIA_F16) {
	Shape inputShape = {2, 5};
	std::vector<float> inputFloat = {-2.0f, -1.2f, -0.5f, 0.4f, 1.6f,
									 2.8f,  -3.0f, 0.0f,  0.9f, -0.1f};
	std::vector<uint16_t> inputData(inputFloat.size());
	std::vector<uint16_t> minData(inputFloat.size(), fp32_to_fp16(-1.0f));
	std::vector<uint16_t> maxData(inputFloat.size(), fp32_to_fp16(1.0f));
	std::transform(inputFloat.begin(), inputFloat.end(), inputData.begin(),
				   [](float v) { return fp32_to_fp16(v); });

	runClipSingleDeviceTest<uint16_t>(INFINI_DEVICE_NVIDIA, inputShape,
									  inputData, minData, maxData,
									  DataType(INFINI_DTYPE_F16), 1e-2f);
}
#endif

} // namespace infini
