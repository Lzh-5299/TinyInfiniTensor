#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        auto output_dim = input_dim;
        int rank = A->getRank();

        IT_ASSERT((int)transposePermute.size() == rank);

        // 检查 perm 合法性（0 ~ rank-1 且不重复）
        std::vector<bool> used(rank, false);
        for (int i = 0; i < rank; ++i)
        {
            int p = transposePermute[i];
            IT_ASSERT(p >= 0 && p < rank);
            IT_ASSERT(!used[p]);
            used[p] = true;
        }

        // 执行维度重排
        for (int i = 0; i < rank; ++i)
        {
            output_dim[i] = input_dim[transposePermute[i]];
        }

        return std::vector<Shape>{output_dim};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
