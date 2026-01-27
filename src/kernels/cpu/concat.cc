#include "operators/concat.h"
#include "core/kernel.h"

namespace infini {

class NaiveConcat : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<ConcatObj>(_op);
        auto inputs = op->getInputs(), outputs = op->getOutputs();
        auto dim = op->getDim();
        auto output = outputs[0];
        std::vector<Shape> iDims;
        for (auto input : inputs)
            iDims.emplace_back(input->getDims());
        const auto &outDim = output->getDims();
        size_t blockOffsetInner = 1;
        for (size_t i = outDim.size() - 1; i > (size_t)dim; --i)
            blockOffsetInner *= outDim[i];
        size_t blockOffset = outDim[dim] * blockOffsetInner;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input = inputs[i];
            auto dimOffset = 0;
            auto iDim = iDims[i];
            for (size_t j = 0; j < i; ++j)
                dimOffset += iDims[j][dim];
            size_t localBlockOffset = 1;
            for (size_t i = iDim.size() - 1;
                 i >= (size_t)dim && i != (size_t)-1; --i)
                localBlockOffset *= iDim[i];
            auto innerOffset = blockOffsetInner * dimOffset;
            auto inSize = input->size();
            auto inPtr = input->getRawDataPtr<T *>(),
                 outPtr = output->getRawDataPtr<T *>();
            
            // 更直观的内存访问模式
            auto outerSize = 1;
            for (size_t i = 0; i < (size_t)dim; ++i)
                outerSize *= iDims[i][i];
            
            auto innerSize = localBlockOffset;
            auto copySize = iDims[i][dim] * blockOffsetInner;
            
#pragma omp parallel for
            for (size_t outer = 0; outer < outerSize; ++outer) {
                for (size_t inner = 0; inner < innerSize; ++inner) {
                    auto inIdx = outer * innerSize + inner;
                    auto outIdx = outer * blockOffset + inner + innerOffset;
                    outPtr[outIdx] = inPtr[inIdx];
                }
            }
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

REGISTER_KERNEL(Device::CPU, OpType::Concat, NaiveConcat, "ConcatNaive_CPU");

} // namespace infini
