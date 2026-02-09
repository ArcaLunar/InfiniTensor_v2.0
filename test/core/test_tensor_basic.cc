#include "core/tensor.h"
#include "gtest/gtest.h"

namespace infini {
class TensorBasicTest : public testing::Test {
  protected:
    void SetUp() override {}
};

// Test constructing Tensor from constant shape vector
TEST_F(TensorBasicTest, ConstructFromConcreteShapeVector) {
    Shape concreteDims = {2, 3, 4};
    auto tensor = make_ref<TensorObj>(concreteDims, DataType(INFINI_DTYPE_F32));

    EXPECT_EQ(tensor->getDataType(), DataType(INFINI_DTYPE_F32));
    EXPECT_EQ(tensor->getRank(), 3);

    auto shape = tensor->getShape();
    EXPECT_TRUE(shape->isConcrete());
    EXPECT_EQ(shape->size(), 3);

    auto constantShape = shape->getConstantValue();
    EXPECT_EQ(constantShape, concreteDims);
}

// Test constructing Tensor from symbolic shape
TEST_F(TensorBasicTest, ConstructFromSymbolicShape) {
    auto batch = ExprObj::variable("batch");
    auto height = ExprObj::constant(224);
    auto width = ExprObj::constant(224);
    auto channels = ExprObj::constant(3);

    auto shapeExpr =
        ShapeExpr(new ShapeExprObj({batch, height, width, channels}));
    auto tensor = make_ref<TensorObj>(shapeExpr, DataType(INFINI_DTYPE_F16));

    EXPECT_EQ(tensor->getDataType(), DataType(INFINI_DTYPE_F16));
    EXPECT_EQ(tensor->getRank(), 4);
    EXPECT_FALSE(tensor->getShape()->isConcrete());
    EXPECT_TRUE(tensor->getShape()->isDynamic());

    EXPECT_EQ(tensor->getShape()->toString(), "[batch, 224, 224, 3]");
}

// Test constructing Tensor from mixed shape
TEST_F(TensorBasicTest, ConstructFromMixedShape) {
    auto shapeExpr = ShapeExpr(new ShapeExprObj(
        {ExprObj::variable("batch"),
         ExprObj::constant(256) + ExprObj::constant(2), // Compute expression
         ExprObj::constant(128), ExprObj::variable("channels")}));

    auto tensor = make_ref<TensorObj>(shapeExpr, DataType(INFINI_DTYPE_I32));

    EXPECT_EQ(tensor->getDataType(), DataType(INFINI_DTYPE_I32));
    EXPECT_EQ(tensor->getRank(), 4);
    EXPECT_EQ(tensor->getShape()->toString(),
              "[batch, (256 + 2), 128, channels]");
}

// Test constructing Tensor with stride
TEST_F(TensorBasicTest, ConstructWithStride) {
    Shape concreteDims = {2, 3, 4};
    Stride strideDims = {12, 4, 1}; // Contiguous stride

    // Construct from constant stride
    auto tensor1 = make_ref<TensorObj>(concreteDims, strideDims,
                                       DataType(INFINI_DTYPE_U32));
    EXPECT_EQ(tensor1->getRank(), 3);

    // Construct from stride expression
    auto strideExpr = StrideExpr(new StrideExprObj(
        {ExprObj::constant(12), ExprObj::constant(4), ExprObj::constant(1)}));

    auto tensor2 = make_ref<TensorObj>(concreteDims, strideExpr,
                                       DataType(INFINI_DTYPE_U32));
    EXPECT_EQ(tensor2->getStride()->toString(), "[12, 4, 1]");
}

// Test shape getter and setter
TEST_F(TensorBasicTest, ShapeGetSet) {
    auto tensor =
        make_ref<TensorObj>(Shape{2, 3, 4}, DataType(INFINI_DTYPE_F32));

    // Get shape
    auto shape = tensor->getShape();
    EXPECT_TRUE(shape->isConcrete());
    EXPECT_EQ(shape->getConstantValue(), Shape({2, 3, 4}));

    // Set to new constant shape
    tensor->setShape(Shape{5, 6, 7});
    auto newShape = tensor->getShape();
    EXPECT_EQ(newShape->getConstantValue(), Shape({5, 6, 7}));

    // Set to symbolic shape
    auto symShape = ShapeExpr(
        new ShapeExprObj({ExprObj::variable("batch"), ExprObj::constant(224),
                          ExprObj::constant(224)}));
    tensor->setShape(symShape);
    EXPECT_FALSE(tensor->getShape()->isConcrete());
    EXPECT_EQ(tensor->getShape()->toString(), "[batch, 224, 224]");
}

// Test stride getter and setter
TEST_F(TensorBasicTest, StrideGetSet) {
    auto tensor =
        make_ref<TensorObj>(Shape{2, 3, 4}, DataType(INFINI_DTYPE_F32));

    // Initial should be contiguous stride
    auto stride = tensor->getStride();
    EXPECT_TRUE(stride->isConcrete());
    // Contiguous stride should be [3*4, 4, 1] = [12, 4, 1]

    // Set new constant stride
    tensor->setStride(Stride{6, 2, 1});
    auto newStride = tensor->getStride();
    EXPECT_EQ(newStride->getConstantValue(), Stride({6, 2, 1}));

    // Set to stride expression
    auto strideExpr = StrideExpr(
        new StrideExprObj({ExprObj::variable("stride0"), ExprObj::constant(2),
                           ExprObj::constant(1)}));
    tensor->setStride(strideExpr);
    EXPECT_EQ(tensor->getStride()->toString(), "[stride0, 2, 1]");
}

// Test total element count calculation
TEST_F(TensorBasicTest, ElementCount) {
    // Constant shape
    auto tensor1 =
        make_ref<TensorObj>(Shape{2, 3, 4}, DataType(INFINI_DTYPE_F32));
    EXPECT_EQ(tensor1->getElement(), 2 * 3 * 4); // 24

    // Symbolic shape
    auto symShape = ShapeExpr(
        new ShapeExprObj({ExprObj::variable("batch"), ExprObj::constant(224),
                          ExprObj::constant(224)}));
    auto tensor2 = make_ref<TensorObj>(symShape, DataType(INFINI_DTYPE_F32));

    EXPECT_THROW(tensor2->getElement(),
                 Exception); // Cannot calculate element count
    tensor2->setShape(symShape->evaluate({{"batch", 3}}).value());
    EXPECT_EQ(tensor2->getElement(), 3 * 224 * 224); // 3*224*224
}

// Test storage size and total bytes
TEST_F(TensorBasicTest, StorageSizeAndBytes) {
    auto tensor =
        make_ref<TensorObj>(Shape{2, 3, 4}, DataType(INFINI_DTYPE_F32));

    EXPECT_EQ(tensor->getElement(), 24);     // 2*3*4
    EXPECT_EQ(tensor->getStorageSize(), 24); // Assume contiguous storage

    // Test total bytes
    auto totalBytes = tensor->getTotalBytes();
    EXPECT_EQ(totalBytes, 24 * sizeof(float));

    // Test different data types
    auto tensorInt8 =
        make_ref<TensorObj>(Shape{10, 20}, DataType(INFINI_DTYPE_I8));
    EXPECT_EQ(tensorInt8->getTotalBytes(), 10 * 20 * sizeof(int8_t));

    auto tensorDouble =
        make_ref<TensorObj>(Shape{5, 5}, DataType(INFINI_DTYPE_F64));
    EXPECT_EQ(tensorDouble->getTotalBytes(), 25 * sizeof(double));
}
} // namespace infini
