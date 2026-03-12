#include "core/runtime.h"
#include "operators/Clip.h"
#include "gtest/gtest.h"

namespace infini {

class ClipBasicTest : public testing::Test {
  protected:
	Runtime runtime;
	Graph graph;

	void SetUp() override {
		runtime = make_ref<RuntimeObj>();
		graph = make_ref<GraphObj>(runtime);
	}
};

TEST_F(ClipBasicTest, BasicConstruction) {
	auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
	auto minVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
	auto maxVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

	auto clip = graph->addOp<ClipObj>(input, minVal, maxVal, nullptr);

	EXPECT_EQ(clip->getOpType(), OpType::Clip);
	EXPECT_EQ(clip->getNumInputs(), 3);
	EXPECT_EQ(clip->getNumOutputs(), 1);
	EXPECT_EQ(clip->getOutDType(0), DataType(INFINI_DTYPE_F32));
}

TEST_F(ClipBasicTest, ShapeInferenceConcrete) {
	auto input = graph->addTensor({4, 5, 6}, DataType(INFINI_DTYPE_F32));
	auto minVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
	auto maxVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

	auto clip = graph->addOp<ClipObj>(input, minVal, maxVal, nullptr);

	auto inferredShapes = clip->inferShape();
	ASSERT_TRUE(inferredShapes.has_value());
	ASSERT_EQ(inferredShapes->size(), 1);

	auto outputShape = (*inferredShapes)[0];
	EXPECT_TRUE(outputShape->isConcrete());
	EXPECT_EQ(outputShape->size(), 3);

	auto shapeValues = outputShape->getConstantValue();
	EXPECT_EQ(shapeValues[0], 4);
	EXPECT_EQ(shapeValues[1], 5);
	EXPECT_EQ(shapeValues[2], 6);
}

TEST_F(ClipBasicTest, SymbolicShapeInference) {
	auto batch = ExprObj::variable("batch");
	auto seq = ExprObj::variable("seq");
	auto hidden = ExprObj::constant(128);

	auto inputShape = ShapeExpr(new ShapeExprObj({batch, seq, hidden}));
	auto input = graph->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
	auto minVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
	auto maxVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

	auto clip = graph->addOp<ClipObj>(input, minVal, maxVal, nullptr);
	auto inferredShapes = clip->inferShape();

	ASSERT_TRUE(inferredShapes.has_value());
	auto outputShape = (*inferredShapes)[0];
	EXPECT_FALSE(outputShape->isConcrete());
	EXPECT_EQ(outputShape->size(), 3);
	EXPECT_EQ(outputShape->toString(), "[batch, seq, 128]");
}

TEST_F(ClipBasicTest, ExplicitOutputTensor) {
	auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
	auto minVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
	auto maxVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
	auto output = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));

	auto clip = graph->addOpWithOutputs<ClipObj>(input, minVal, maxVal, output);

	EXPECT_EQ(clip->getOutput(0), output);
}

TEST_F(ClipBasicTest, MinMaxDataTypeMismatch) {
	auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
	auto minVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F16));
	auto maxVal = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

	EXPECT_THROW(graph->addOp<ClipObj>(input, minVal, maxVal, nullptr),
				 Exception);
}

} // namespace infini
