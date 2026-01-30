#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB), m(0), n(0), k(0)
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
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        
        size_t rankA = shapeA.size();
        size_t rankB = shapeB.size();
        
        // 至少需要是 2D 张量
        if (rankA < 2 || rankB < 2) {
            return std::nullopt;
        }
        
        // 获取 A 的 m 和 k 维度（考虑转置）
        int m_val, k_from_A;
        if (transA) {
            k_from_A = shapeA[rankA - 2];
            m_val = shapeA[rankA - 1];
        } else {
            m_val = shapeA[rankA - 2];
            k_from_A = shapeA[rankA - 1];
        }
        
        // 获取 B 的 k 和 n 维度（考虑转置）
        int k_from_B, n_val;
        if (transB) {
            n_val = shapeB[rankB - 2];
            k_from_B = shapeB[rankB - 1];
        } else {
            k_from_B = shapeB[rankB - 2];
            n_val = shapeB[rankB - 1];
        }
        
        // 验证 k 维度匹配
        if (k_from_A != k_from_B) {
            return std::nullopt;
        }
        
        // 设置成员变量 m, n, k
        // 注意：这里需要用 const_cast 因为 inferShape 是 const 方法
        // 或者更好的做法是在其他地方设置
        const_cast<MatmulObj*>(this)->m = m_val;
        const_cast<MatmulObj*>(this)->n = n_val;
        const_cast<MatmulObj*>(this)->k = k_from_A;
        
        // 构建输出 shape
        Shape output_shape;
        
        // 处理 batch 维度的广播
        size_t maxRank = std::max(rankA, rankB);
        
        // 处理 batch 维度（除了最后两维）
        for (size_t i = 0; i < maxRank - 2; ++i) {
            int dimA = 1, dimB = 1;
            
            // 获取 A 的对应 batch 维度
            if (i + rankA >= maxRank) {
                dimA = shapeA[i + rankA - maxRank];
            }
            
            // 获取 B 的对应 batch 维度
            if (i + rankB >= maxRank) {
                dimB = shapeB[i + rankB - maxRank];
            }
            
            // 广播规则检查
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                return std::nullopt;
            }
            
            output_shape.push_back(std::max(dimA, dimB));
        }
        
        // 添加矩阵乘法的输出维度 [m, n]
        output_shape.push_back(m_val);
        output_shape.push_back(n_val);
        
        return {{output_shape}};
    }

} // namespace infini