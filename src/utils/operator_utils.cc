#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    
    // 结果的维度数是两者中较大的
    size_t rankA = A.size();
    size_t rankB = B.size();
    size_t rankMax = std::max(rankA, rankB);
    
    Shape result(rankMax);
    
    // 从右向左对齐，逐维度比较
    for (size_t i = 0; i < rankMax; ++i) {
        // 从右往左的索引
        // 如果某个形状维度不够，视为1
        int dimA = (i < rankA) ? A[rankA - 1 - i] : 1;
        int dimB = (i < rankB) ? B[rankB - 1 - i] : 1;
        
        // 广播规则：两个维度必须相等，或者其中一个为1
        if (dimA == dimB) {
            result[rankMax - 1 - i] = dimA;
        } else if (dimA == 1) {
            result[rankMax - 1 - i] = dimB;
        } else if (dimB == 1) {
            result[rankMax - 1 - i] = dimA;
        } else {
            // 不兼容的形状，理论上不应该到这里（checkValid会先检查）
            IT_ASSERT(false);  // 广播失败
        }
    }
    
    return result;
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