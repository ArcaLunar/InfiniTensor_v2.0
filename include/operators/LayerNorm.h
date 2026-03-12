#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/layer_norm.h>

namespace infini {

class LayerNormObj : public OperatorObj {
  private:
    float eps;

  public:
    // inputs = {input, weight, bias}, outputs = {y}
    // LayerNorm is applied on the last dimension by default.
    LayerNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor bias,
                 Tensor output, float eps = 1e-5f);
    string toString() const override;
    ~LayerNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    float getEps() const;
};

} // namespace infini