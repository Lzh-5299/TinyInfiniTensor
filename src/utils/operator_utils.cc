#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // 从尾部对齐
    int rankA = A.size();
    int rankB = B.size();
    int outRank = std::max(rankA, rankB);

    Shape out(outRank, 1);

    for (int i = 0; i < outRank; ++i) {
        // 从最后一维开始对齐
        int idxA = rankA - 1 - i;
        int idxB = rankB - 1 - i;

        size_t dimA = (idxA >= 0) ? A[idxA] : 1;
        size_t dimB = (idxB >= 0) ? B[idxB] : 1;

        // 广播规则
        if (dimA == dimB || dimA == 1) {
            out[outRank - 1 - i] = dimB;
        } else if (dimB == 1) {
            out[outRank - 1 - i] = dimA;
        } else {
            // 不可广播
            IT_ASSERT(false && "Broadcast shape mismatch");
        }
    }

    return out;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
