#include "core/graph.h"
#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "gtest/gtest.h"

namespace infini {
class GraphOptimizerTest : public testing::Test {
  protected:
    Runtime runtime;

    void SetUp() override { runtime = make_ref<RuntimeObj>(); }

    Tensor makeScalar(Graph graph, float value) {
        auto tensor = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
        auto *buffer = new float[1];
        buffer[0] = value;
        tensor->setData(buffer);
        return tensor;
    }
};

TEST_F(GraphOptimizerTest, IdentityEliminationPreservesOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto zero = makeScalar(graph, 0.0f);
    auto add = graph->addOp<ElementWiseObj>(OpType::Add, input, zero, nullptr);
    graph->setOutputs({add->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, ConstantFoldAddProducesConstantOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto lhs = makeScalar(graph, 2.0f);
    auto rhs = makeScalar(graph, 3.0f);
    auto add = graph->addOp<ElementWiseObj>(OpType::Add, lhs, rhs, nullptr);
    auto output = add->getOutput(0);
    graph->setOutputs({output});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    EXPECT_TRUE(output->isConstant());
    ASSERT_NE(output->getData(), nullptr);
    EXPECT_FLOAT_EQ(output->getRawDataPtr<float *>()[0], 5.0f);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, DeadCodeEliminationDropsUnusedBranch) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto one = makeScalar(graph, 1.0f);
    auto two = makeScalar(graph, 2.0f);
    auto live = graph->addOp<ElementWiseObj>(OpType::Mul, input, one, nullptr);
    auto dead = graph->addOp<ElementWiseObj>(OpType::Mul, input, two, nullptr);
    auto deadOutputFuid = dead->getOutput(0)->getFuid();
    graph->setOutputs({live->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_EQ(graph->getTensor(deadOutputFuid), nullptr);
    EXPECT_TRUE(graph->checkValid());
}
} // namespace infini
