#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LayerNormBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LayerNormBasicTest, BasicConstruction) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));

    auto layernorm =
        graph->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

    EXPECT_EQ(layernorm->getOpType(), OpType::LayerNorm);
    EXPECT_EQ(layernorm->getNumInputs(), 3);
    EXPECT_EQ(layernorm->getNumOutputs(), 1);
    EXPECT_FLOAT_EQ(layernorm->getEps(), 1e-5f);
}

TEST_F(LayerNormBasicTest, ShapeInferenceConcrete) {
    auto input = graph->addTensor({2, 3, 8}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));

    auto layernorm =
        graph->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

    auto inferredShapes = layernorm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);

    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 8);
}

TEST_F(LayerNormBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto seq = ExprObj::variable("seq");
    auto hidden = ExprObj::constant(128);

    auto inputShape = ShapeExpr(new ShapeExprObj({batch, seq, hidden}));
    auto input = graph->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({128}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({128}, DataType(INFINI_DTYPE_F32));

    auto layernorm =
        graph->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

    auto inferredShapes = layernorm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto outputShape = (*inferredShapes)[0];

    EXPECT_FALSE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);
    EXPECT_EQ(outputShape->toString(), "[batch, seq, 128]");
}

TEST_F(LayerNormBasicTest, DataTypeInferenceF32) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));

    auto layernorm =
        graph->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

    auto inferredTypes = layernorm->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(LayerNormBasicTest, DataTypeInferenceF16) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F16));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F16));
    auto bias = graph->addTensor({4}, DataType(INFINI_DTYPE_F16));

    auto layernorm =
        graph->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

    auto inferredTypes = layernorm->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F16));
}

TEST_F(LayerNormBasicTest, DataTypeInferenceBF16) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_BF16));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_BF16));
    auto bias = graph->addTensor({4}, DataType(INFINI_DTYPE_BF16));

    auto layernorm =
        graph->addOp<LayerNormObj>(input, weight, bias, nullptr, 1e-5f);

    auto inferredTypes = layernorm->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_BF16));
}

TEST_F(LayerNormBasicTest, ExplicitOutputTensor) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto output = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));

    auto layernorm =
        graph->addOpWithOutputs<LayerNormObj>(input, weight, bias, output,
                                              1e-5f);

    EXPECT_EQ(layernorm->getOutput(0), output);
}

TEST_F(LayerNormBasicTest, ExplicitOutputShapeMismatch) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto wrongOutput = graph->addTensor({2, 5}, DataType(INFINI_DTYPE_F32));

    EXPECT_THROW(graph->addOpWithOutputs<LayerNormObj>(input, weight, bias,
                                                       wrongOutput, 1e-5f),
                 Exception);
}

} // namespace infini
