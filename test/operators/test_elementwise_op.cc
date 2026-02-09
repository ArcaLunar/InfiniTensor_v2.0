#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "gtest/gtest.h"

namespace infini {

class ElementWiseBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// Test basic construction of ElementWise
TEST_F(ElementWiseBasicTest, BasicConstruction) {
    auto A = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);
    EXPECT_EQ(elementwise->getOpType(), OpType::Add);
    EXPECT_EQ(elementwise->getNumInputs(), 2);
    EXPECT_EQ(elementwise->getNumOutputs(), 1);
    EXPECT_EQ(elementwise->getElemenwiseOpType(), OpType::Add);
}

// Test ElementWise shape inference - same shape
TEST_F(ElementWiseBasicTest, ShapeInferenceSameShape) {
    auto A = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());

    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues.size(), 3);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// Test ElementWise shape inference - broadcast (scalar to tensor)
TEST_F(ElementWiseBasicTest, ShapeInferenceScalarBroadcast) {
    // Scalar broadcasts to [2, 3, 4]
    auto scalar = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto tensor = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));

    auto elementwise =
        graph->addOp<ElementWiseObj>(OpType::Mul, scalar, tensor, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();

    EXPECT_EQ(shapeValues.size(),
              3); // Scalar should broadcast to tensor's dimension
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// Test ElementWise shape inference - broadcast (both operands need broadcast)
TEST_F(ElementWiseBasicTest, ShapeInferenceBothBroadcast) {
    // [1, 3, 1] and [2, 1, 4] broadcast to [2, 3, 4]
    auto A = graph->addTensor({1, 3, 1}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 1, 4}, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Mul, A, B, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();

    EXPECT_EQ(shapeValues.size(), 3);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// Test ElementWise data type inference
TEST_F(ElementWiseBasicTest, DataTypeInference) {
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    auto inferredTypes = elementwise->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

// Test symbolic shape inference
TEST_F(ElementWiseBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto height = ExprObj::variable("h");
    auto width = ExprObj::constant(256);

    auto shapeA = ShapeExpr(new ShapeExprObj({batch, height, width}));
    auto shapeB = ShapeExpr(new ShapeExprObj({height, width}));

    auto A = graph->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor(shapeB, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    EXPECT_FALSE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);

    // Check symbolic expression
    EXPECT_EQ(outputShape->toString(), "[batch, h, 256]");
}

} // namespace infini
