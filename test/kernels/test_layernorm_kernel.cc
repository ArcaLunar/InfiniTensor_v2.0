#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

#include <cmath>
#include <thread>
#include <type_traits>

namespace infini {

template <typename T> struct LayerNormThreadTestParams {
	infiniDevice_t device = INFINI_DEVICE_CPU;
	int deviceId = 0;
	Shape inputShape;
	DataType dataType = DataType(INFINI_DTYPE_F32);
	float eps = 1e-5f;
	std::vector<T> inputData;
	std::vector<T> weightData;
	std::vector<T> biasData;
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
std::vector<float> layerNormReference(const Shape &inputShape,
									  const std::vector<T> &inputData,
									  const std::vector<T> &weightData,
									  const std::vector<T> &biasData,
									  float eps) {
	IT_ASSERT(inputShape.size() >= 2,
			  "LayerNorm input rank must be at least 2 in tests");

	size_t normSize = inputShape.back();
	size_t outer = 1;
	for (size_t i = 0; i + 1 < inputShape.size(); ++i)
		outer *= inputShape[i];

	IT_ASSERT(inputData.size() == outer * normSize,
			  "Input data size mismatch");
	IT_ASSERT(weightData.size() == normSize,
			  "Weight data size mismatch with normalized dimension");
	IT_ASSERT(biasData.size() == normSize,
			  "Bias data size mismatch with normalized dimension");

	std::vector<float> ref(inputData.size(), 0.0f);
	for (size_t o = 0; o < outer; ++o) {
		float mean = 0.0f;
		for (size_t j = 0; j < normSize; ++j) {
			mean += toFloatValue(inputData[o * normSize + j]);
		}
		mean /= static_cast<float>(normSize);

		float var = 0.0f;
		for (size_t j = 0; j < normSize; ++j) {
			float x = toFloatValue(inputData[o * normSize + j]);
			float d = x - mean;
			var += d * d;
		}
		var /= static_cast<float>(normSize);
		float invStd = 1.0f / std::sqrt(var + eps);

		for (size_t j = 0; j < normSize; ++j) {
			float x = toFloatValue(inputData[o * normSize + j]);
			float w = toFloatValue(weightData[j]);
			float b = toFloatValue(biasData[j]);
			ref[o * normSize + j] = (x - mean) * invStd * w + b;
		}
	}
	return ref;
}

template <typename T>
void layerNormDeviceThreadFunc(LayerNormThreadTestParams<T> &params) {
	RuntimeObj::init();
	Runtime &runtime = RuntimeObj::getInstance();

	runtime->initThreadContext(params.device, params.deviceId);

	size_t normSize = params.inputShape.back();
	Graph g = make_ref<GraphObj>(runtime);
	auto input = g->addTensor(params.inputShape, params.dataType);
	auto weight = g->addTensor({normSize}, params.dataType);
	auto bias = g->addTensor({normSize}, params.dataType);
	auto op = g->addOp<LayerNormObj>(input, weight, bias, nullptr, params.eps);

	input->setData(params.inputData.data());
	weight->setData(params.weightData.data());
	bias->setData(params.biasData.data());
	runtime->dataMalloc(g);
	runtime->run(g);

	auto output = op->getOutput(0);
	size_t numElements = output->getElement();
	params.outputData.resize(numElements);

	void *hostPtr = runtime->allocHost(output->getTotalBytes());
	runtime->memcpy(hostPtr, output->template getRawDataPtr<void *>(),
					output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
	copyAndConvertData(params.outputData, hostPtr, numElements,
					   params.dataType);
	runtime->deallocHost(hostPtr);

	params.completed = true;
}

template <typename T>
void runLayerNormMultiThreadTest(const Shape &inputShape,
								 const std::vector<T> &inputData,
								 const std::vector<T> &weightData,
								 const std::vector<T> &biasData,
								 const DataType &dataType, float eps,
								 float tol) {
	LayerNormThreadTestParams<T> cpuParams, gpuParams;

	cpuParams.device = INFINI_DEVICE_CPU;
	cpuParams.inputShape = inputShape;
	cpuParams.dataType = dataType;
	cpuParams.eps = eps;
	cpuParams.inputData = inputData;
	cpuParams.weightData = weightData;
	cpuParams.biasData = biasData;

	gpuParams.device = INFINI_DEVICE_NVIDIA;
	gpuParams.inputShape = inputShape;
	gpuParams.dataType = dataType;
	gpuParams.eps = eps;
	gpuParams.inputData = inputData;
	gpuParams.weightData = weightData;
	gpuParams.biasData = biasData;

	std::thread cpuThread(layerNormDeviceThreadFunc<T>, std::ref(cpuParams));
	std::thread gpuThread(layerNormDeviceThreadFunc<T>, std::ref(gpuParams));
	cpuThread.join();
	gpuThread.join();

	ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
	ASSERT_TRUE(gpuParams.completed) << "NVIDIA thread failed";
	ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size());

	auto ref = layerNormReference(inputShape, inputData, weightData, biasData,
								  eps);

	size_t numErrors = 0;
	for (size_t i = 0; i < ref.size(); ++i) {
		float cpuVal = toFloatValue(cpuParams.outputData[i]);
		float gpuVal = toFloatValue(gpuParams.outputData[i]);
		float refVal = ref[i];
		if (std::abs(cpuVal - gpuVal) > tol || std::abs(cpuVal - refVal) > tol ||
			std::abs(gpuVal - refVal) > tol) {
			++numErrors;
			if (numErrors <= 5) {
				std::cout << "Mismatch at index " << i << ": ref=" << refVal
						  << ", cpu=" << cpuVal << ", gpu=" << gpuVal
						  << std::endl;
			}
		}
	}
	EXPECT_EQ(numErrors, 0) << "LayerNorm CPU/GPU mismatch";
}

template <typename T>
void runLayerNormSingleDeviceTest(infiniDevice_t device, const Shape &inputShape,
								  const std::vector<T> &inputData,
								  const std::vector<T> &weightData,
								  const std::vector<T> &biasData,
								  const DataType &dataType, float eps,
								  float tol) {
	RuntimeObj::init();
	Runtime &runtime = RuntimeObj::getInstance();
	runtime->initThreadContext(device, 0);

	size_t normSize = inputShape.back();
	Graph g = make_ref<GraphObj>(runtime);
	auto input = g->addTensor(inputShape, dataType);
	auto weight = g->addTensor({normSize}, dataType);
	auto bias = g->addTensor({normSize}, dataType);
	auto op = g->addOp<LayerNormObj>(input, weight, bias, nullptr, eps);

	input->setData(const_cast<T *>(inputData.data()));
	weight->setData(const_cast<T *>(weightData.data()));
	bias->setData(const_cast<T *>(biasData.data()));
	runtime->dataMalloc(g);
	runtime->run(g);

	auto output = op->getOutput(0);
	std::vector<T> outputData(output->getElement());
	void *hostPtr = runtime->allocHost(output->getTotalBytes());
	runtime->memcpy(hostPtr, output->template getRawDataPtr<void *>(),
					output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
	copyAndConvertData(outputData, hostPtr, output->getElement(), dataType);
	runtime->deallocHost(hostPtr);

	auto ref = layerNormReference(inputShape, inputData, weightData, biasData,
								  eps);
	for (size_t i = 0; i < ref.size(); ++i) {
		EXPECT_NEAR(toFloatValue(outputData[i]), ref[i], tol);
	}
}

TEST(LayerNorm, Basic_MultiThread_F32) {
	Shape inputShape = {2, 4};
	std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f,
									2.0f, 4.0f, 6.0f, 8.0f};
	std::vector<float> weightData = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> biasData = {0.0f, 0.0f, 0.0f, 0.0f};

#ifdef USE_CUDA
	runLayerNormMultiThreadTest<float>(
		inputShape, inputData, weightData, biasData,
		DataType(INFINI_DTYPE_F32), 1e-5f, 1e-3f);
#else
	std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

TEST(LayerNorm, Basic_MultiThread_F16) {
	Shape inputShape = {2, 4};
	std::vector<float> inputFloat = {1.0f, 2.0f, 3.0f, 4.0f,
									 2.0f, 4.0f, 6.0f, 8.0f};
	std::vector<float> weightFloat = {1.0f, 0.5f, 1.5f, 1.0f};
	std::vector<float> biasFloat = {0.1f, -0.2f, 0.3f, 0.0f};

	std::vector<uint16_t> inputData(inputFloat.size());
	std::vector<uint16_t> weightData(weightFloat.size());
	std::vector<uint16_t> biasData(biasFloat.size());
	for (size_t i = 0; i < inputFloat.size(); ++i)
		inputData[i] = fp32_to_fp16(inputFloat[i]);
	for (size_t i = 0; i < weightFloat.size(); ++i)
		weightData[i] = fp32_to_fp16(weightFloat[i]);
	for (size_t i = 0; i < biasFloat.size(); ++i)
		biasData[i] = fp32_to_fp16(biasFloat[i]);

#ifdef USE_CUDA
	runLayerNormMultiThreadTest<uint16_t>(
		inputShape, inputData, weightData, biasData,
		DataType(INFINI_DTYPE_F16), 1e-5f, 2e-2f);
#else
	std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

TEST(LayerNorm, SingleDevice_CPU_F32) {
	Shape inputShape = {2, 3, 4};
	std::vector<float> inputData = {
		1.0f, 2.0f, 3.0f, 4.0f,
		2.0f, 3.0f, 4.0f, 5.0f,
		3.0f, 4.0f, 5.0f, 6.0f,
		4.0f, 5.0f, 6.0f, 7.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		6.0f, 7.0f, 8.0f, 9.0f,
	};
	std::vector<float> weightData = {1.0f, 0.8f, 1.2f, 1.0f};
	std::vector<float> biasData = {0.0f, 0.1f, -0.1f, 0.0f};

	runLayerNormSingleDeviceTest<float>(
		INFINI_DEVICE_CPU, inputShape, inputData, weightData, biasData,
		DataType(INFINI_DTYPE_F32), 1e-5f, 1e-3f);
}

#ifdef USE_CUDA
TEST(LayerNorm, SingleDevice_NVIDIA_F32) {
	Shape inputShape = {2, 4};
	std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f,
									2.0f, 4.0f, 6.0f, 8.0f};
	std::vector<float> weightData = {1.0f, 1.0f, 1.0f, 1.0f};
	std::vector<float> biasData = {0.0f, 0.0f, 0.0f, 0.0f};

	runLayerNormSingleDeviceTest<float>(
		INFINI_DEVICE_NVIDIA, inputShape, inputData, weightData, biasData,
		DataType(INFINI_DTYPE_F32), 1e-5f, 1e-3f);
}

TEST(LayerNorm, SingleDevice_NVIDIA_F16) {
	Shape inputShape = {2, 4};
	std::vector<float> inputFloat = {1.0f, 2.0f, 3.0f, 4.0f,
									 2.0f, 4.0f, 6.0f, 8.0f};
	std::vector<float> weightFloat = {1.0f, 0.5f, 1.5f, 1.0f};
	std::vector<float> biasFloat = {0.1f, -0.2f, 0.3f, 0.0f};

	std::vector<uint16_t> inputData(inputFloat.size());
	std::vector<uint16_t> weightData(weightFloat.size());
	std::vector<uint16_t> biasData(biasFloat.size());
	for (size_t i = 0; i < inputFloat.size(); ++i)
		inputData[i] = fp32_to_fp16(inputFloat[i]);
	for (size_t i = 0; i < weightFloat.size(); ++i)
		weightData[i] = fp32_to_fp16(weightFloat[i]);
	for (size_t i = 0; i < biasFloat.size(); ++i)
		biasData[i] = fp32_to_fp16(biasFloat[i]);

	runLayerNormSingleDeviceTest<uint16_t>(
		INFINI_DEVICE_NVIDIA, inputShape, inputData, weightData, biasData,
		DataType(INFINI_DTYPE_F16), 1e-5f, 2e-2f);
}
#endif

TEST(LayerNorm, WeightShapeMismatchThrows) {
	RuntimeObj::init();
	Runtime &runtime = RuntimeObj::getInstance();
	runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

	Shape inputShape = {2, 4};
	Graph g = make_ref<GraphObj>(runtime);
	auto input = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
	auto weight = g->addTensor({3}, DataType(INFINI_DTYPE_F32));
	auto bias = g->addTensor({4}, DataType(INFINI_DTYPE_F32));
	g->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

	std::vector<float> inputData = {1, 2, 3, 4, 2, 3, 4, 5};
	std::vector<float> weightData = {1, 1, 1};
	std::vector<float> biasData = {0, 0, 0, 0};
	input->setData(inputData.data());
	weight->setData(weightData.data());
	bias->setData(biasData.data());
	runtime->dataMalloc(g);

	EXPECT_THROW(runtime->run(g), Exception);
}

TEST(LayerNorm, BiasShapeMismatchThrows) {
	RuntimeObj::init();
	Runtime &runtime = RuntimeObj::getInstance();
	runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

	Shape inputShape = {2, 4};
	Graph g = make_ref<GraphObj>(runtime);
	auto input = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
	auto weight = g->addTensor({4}, DataType(INFINI_DTYPE_F32));
	auto bias = g->addTensor({3}, DataType(INFINI_DTYPE_F32));
	g->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

	std::vector<float> inputData = {1, 2, 3, 4, 2, 3, 4, 5};
	std::vector<float> weightData = {1, 1, 1, 1};
	std::vector<float> biasData = {0, 0, 0};
	input->setData(inputData.data());
	weight->setData(weightData.data());
	bias->setData(biasData.data());
	runtime->dataMalloc(g);

	EXPECT_THROW(runtime->run(g), Exception);
}

} // namespace infini
