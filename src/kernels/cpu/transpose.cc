#include "operators/transpose.h"
#include "core/kernel.h"

namespace infini {

inline Shape idx2Pos(const Shape &shape, size_t idx) {
    Shape pos = Shape(shape.size(), 0);
    auto rest = idx, curDimId = shape.size() - 1;
    while (rest > 0) {
        pos[curDimId] = rest % shape[curDimId];
        rest /= shape[curDimId];
        curDimId--;
    }
    return pos;
}

class NaiveTranspose : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<TransposeObj>(_op);
        auto inputs = op->getInputs(), outputs = op->getOutputs();
        const auto &inDim = inputs[0]->getDims();
        const auto &perm = op->getPermute();

        size_t inSize = inputs[0]->size();
        auto inPtr = inputs[0]->getRawDataPtr<T *>(),
             outPtr = outputs[0]->getRawDataPtr<T *>();
        // #pragma omp parallel for
        // 计算输出维度
        Shape outDim(inDim.size());
        for (size_t i = 0; i < inDim.size(); ++i)
            outDim[i] = inDim[perm[i]];
        
        // 计算输入和输出的strides
        Shape inStride(inDim.size(), 1);
        for (int i = inDim.size() - 2; i >= 0; --i)
            inStride[i] = inStride[i + 1] * inDim[i + 1];
            
        Shape outStride(outDim.size(), 1);
        for (int i = outDim.size() - 2; i >= 0; --i)
            outStride[i] = outStride[i + 1] * outDim[i + 1];
        
        // 更高效的内存访问模式
        #pragma omp parallel for
        for (size_t outIdx = 0; outIdx < inSize; ++outIdx) {
            size_t inIdx = 0;
            size_t tmp = outIdx;
            for (size_t i = 0; i < outDim.size(); ++i) {
                size_t dim = perm[i];
                size_t coord = tmp / outStride[i];
                tmp = tmp % outStride[i];
                inIdx += coord * inStride[dim];
            }
            outPtr[outIdx] = inPtr[inIdx];
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Transpose, NaiveTranspose,
                "TransposeNaive_CPU");

} // namespace infini
