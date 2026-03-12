#include "operators/Clip.h"

namespace infini {

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor min_val, Tensor max_val,
                 Tensor output)
    : OperatorObj(OpType::Clip, {input, min_val, max_val}, {output}) {
    IT_ASSERT(checkValid(graph));
}

string ClipObj::toString() const {
    std::ostringstream os;
    os << "Clip(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "min_val=" << inputs[1]->getGuid() << ",";
    os << "max_val=" << inputs[2]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> ClipObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> ClipObj::inferDataType() const {
    // output dtype follows the input tensor (inputs[0]);
    // min/max tensors are expected to have the same dtype
    IT_ASSERT(inputs[1]->getDataType() == inputs[0]->getDataType());
    IT_ASSERT(inputs[2]->getDataType() == inputs[0]->getDataType());
    return {inputs[0]->getDataType()};
}

ClipObj::~ClipObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyClipDescriptor(
            (infiniopClipDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr
                << "Warning: Clip descriptor destroy failed with error code "
                << err << std::endl;
        }
    }
}

void ClipObj::createOpDesc() {
    // inputs[0]=x, inputs[1]=min_val, inputs[2]=max_val, outputs[0]=y
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    auto minShape = inputs[1]->getShape();
    auto maxShape = inputs[2]->getShape();
    auto minStride = inputs[1]->getStride();
    auto maxStride = inputs[2]->getStride();

    infiniopTensorDescriptor_t yDesc, xDesc, minDesc, maxDesc;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yDesc, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xDesc, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &minDesc, minShape->size(), minShape->getConstantValue().data(),
        minStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &maxDesc, maxShape->size(), maxShape->getConstantValue().data(),
        maxStride->getConstantValue().data(),
        inputs[2]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    // API: infiniopCreateClipDescriptor(handle, &desc, y, x, min_val, max_val)
    CHECK_INFINI_ERROR(infiniopCreateClipDescriptor(
        handle, (infiniopClipDescriptor_t *)&infiniOpDesc, yDesc, xDesc,
        minDesc, maxDesc));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(minDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(maxDesc));
}

} // namespace infini