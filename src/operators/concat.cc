#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(!inputs.empty());

    Shape dims = inputs[0]->getDims();
    int rank = inputs[0]->getRank();

    IT_ASSERT(dim >= 0 && dim < rank);

    // concat 维度累加
    size_t concatSize = 0;

    for (const auto &t : inputs) {
        IT_ASSERT(t->getRank() == rank);
        auto curDims = t->getDims();

        for (int i = 0; i < rank; ++i) {
            if (i == dim)
                continue;
            // 非 concat 维度必须一致
            IT_ASSERT(curDims[i] == dims[i]);
        }

        concatSize += curDims[dim];
    }

    dims[dim] = concatSize;

    return std::vector<Shape>{dims};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
