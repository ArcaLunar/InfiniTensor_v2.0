#include "operators/LayerNorm.h"
#include "core/runtime.h"

namespace infini {

// inputs = {input, weight, bias}, outputs = {y}
LayerNormObj::LayerNormObj(GraphObj *graph, Tensor input, Tensor weight,
                           Tensor bias, Tensor output, int axis, float eps)
    : OperatorObj(OpType::LayerNorm, {input, weight, bias}, {output}),
      axis(axis), eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "LayerNorm(axis=" << axis << ",eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "bias=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> LayerNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> LayerNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

LayerNormObj::~LayerNormObj() {
    if (infiniOpDesc)
        infiniopDestroyLayerNormDescriptor(
            (infiniopLayerNormDescriptor_t)infiniOpDesc);
}

void LayerNormObj::createOpDesc() {
    // inputs[0]=x, inputs[1]=weight, inputs[2]=bias, outputs[0]=y
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    auto bShape = inputs[2]->getShape();
    auto yShape = outputs[0]->getShape();
    auto xStride = inputs[0]->getStride();
    auto wStride = inputs[1]->getStride();
    auto bStride = inputs[2]->getStride();
    auto yStride = outputs[0]->getStride();

    // Mean and std_dev have the shape of the "batch prefix": x.shape[:axis].
    // Compute concrete shape values and contiguous strides for them.
    auto xVals = xShape->getConstantValue();
    IT_ASSERT((int)xVals.size() >= axis,
              "axis exceeds input rank in LayerNorm");
    vector<int64_t> meanShape(xVals.begin(), xVals.begin() + axis);
    // Contiguous strides for meanShape (row-major)
    vector<int64_t> meanStride(axis, 1);
    for (int i = axis - 2; i >= 0; --i)
        meanStride[i] = meanStride[i + 1] * (int64_t)meanShape[i + 1];
    // Convert to types expected by infiniopCreateTensorDescriptor
    vector<size_t> meanShapeSz(meanShape.begin(), meanShape.end());
    vector<ptrdiff_t> meanStrideP(meanStride.begin(), meanStride.end());

    infiniDtype_t dtype = inputs[0]->getDataType().getType();

    infiniopTensorDescriptor_t yDesc, meanDesc, stdDesc, xDesc, wDesc, bDesc;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yDesc, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(), dtype));
    // mean/std descriptors (same shape/stride/dtype as batch prefix)
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &meanDesc, (uint64_t)axis, meanShapeSz.data(),
        meanStrideP.data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &stdDesc, (uint64_t)axis, meanShapeSz.data(),
        meanStrideP.data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xDesc, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wDesc, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &bDesc, bShape->size(), bShape->getConstantValue().data(),
        bStride->getConstantValue().data(),
        inputs[2]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateLayerNormDescriptor(
        handle, (infiniopLayerNormDescriptor_t *)&infiniOpDesc, yDesc, meanDesc,
        stdDesc, xDesc, wDesc, bDesc, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(meanDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(stdDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bDesc));
}

int LayerNormObj::getAxis() const { return axis; }
float LayerNormObj::getEps() const { return eps; }

} // namespace infini
