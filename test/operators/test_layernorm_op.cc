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
    // Normalize over last dim (axis=1 for 2D input)
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 1, 1e-5f);
    EXPECT_EQ(op->getOpType(), OpType::LayerNorm);
    EXPECT_EQ(op->getNumInputs(), 3);
    EXPECT_EQ(op->getNumOutputs(), 1);
    EXPECT_EQ(op->getAxis(), 1);
    EXPECT_FLOAT_EQ(op->getEps(), 1e-5f);
}

TEST_F(LayerNormBasicTest, ShapeInference3D) {
    auto x = graph->addTensor({2, 4, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 1);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2); EXPECT_EQ(vals[1], 4); EXPECT_EQ(vals[2], 8);
}

TEST_F(LayerNormBasicTest, DataTypeInference) {
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 1);
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

} // namespace infini
