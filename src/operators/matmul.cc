#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        const auto B = inputs[1];

        auto shapeA = A->getDims();
        auto shapeB = B->getDims();

        IT_ASSERT(shapeA.size() >= 2);
        IT_ASSERT(shapeB.size() >= 2);

        int rankA = shapeA.size();
        int rankB = shapeB.size();

        // 取矩阵维
        size_t a_m = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        size_t a_k = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];

        size_t b_k = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        size_t b_n = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];

        // K 维必须一致
        IT_ASSERT(a_k == b_k);

        // 保存 mnk（给 toString / kernel 用）
        m = a_m;
        n = b_n;
        k = a_k;

        // batch 维
        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);

        // 广播 batch 维
        Shape batchOut = infer_broadcast(batchA, batchB);

        // 拼接输出形状
        Shape out = batchOut;
        out.push_back(m);
        out.push_back(n);

        return std::vector<Shape>{out};
    }

} // namespace infini