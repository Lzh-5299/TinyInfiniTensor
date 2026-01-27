#include "core/graph.h"
#include "utils/operator_utils.h"
#include "core/blob.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{
    static bool isTranspose(const Operator &op)
    {
        return op->getOpType() == OpType::Transpose;
    }

    static bool isMatMul(const Operator &op)
    {
        return op->getOpType() == OpType::MatMul;
    }

    // 判断是不是“最后两个维度交换”
    static bool isSwapLast2Dims(const Operator &op)
    {
        // Transpose操作符直接使用getPermute方法获取permute属性
        auto transOp = std::dynamic_pointer_cast<TransposeObj>(op);
        if (!transOp) return false;
        
        auto perm = transOp->getPermute();
        int n = perm.size();
        if (n < 2)
            return false;
        for (int i = 0; i < n - 2; ++i)
            if (perm[i] != i)
                return false;
        return perm[n - 1] == n - 2 && perm[n - 2] == n - 1;
    }

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        bool changed = true;

        while (changed)
        {
            changed = false;

            // ===============================
            // Pass 1: 消除相邻反向 Transpose
            // ===============================
            // 使用 while 循环和显式的迭代器控制
            auto it = ops.begin();
            while (it != ops.end())
            {
                auto op = *it;
                if (!isTranspose(op))
                {
                    ++it;
                    continue;
                }

                auto out = op->getOutputs()[0];
                if (out->getTargets().size() != 1)
                {
                    ++it;
                    continue;
                }

                auto next = out->getTargets()[0];
                if (!isTranspose(next))
                {
                    ++it;
                    continue;
                }

                // op -> tensor -> next
                auto transOp1 = std::dynamic_pointer_cast<TransposeObj>(op);
                auto transOp2 = std::dynamic_pointer_cast<TransposeObj>(next);
                if (!transOp1 || !transOp2)
                {
                    ++it;
                    continue;
                }
                
                auto perm1 = transOp1->getPermute();
                auto perm2 = transOp2->getPermute();

                // 判断是否互逆
                bool inverse = true;
                for (int i = 0; i < (int)perm1.size(); ++i)
                {
                    if (perm2[perm1[i]] != i)
                    {
                        inverse = false;
                        break;
                    }
                }
                if (!inverse)
                {
                    ++it;
                    continue;
                }

                // 重连图
                auto inTensor = op->getInputs()[0];
                auto outTensor = next->getOutputs()[0];

                // 所有使用 next 输出的地方，改成用 op 输入
                for (auto &succ : outTensor->getTargets())
                {
                    succ->replaceInput(outTensor, inTensor);
                    inTensor->addTarget(succ);
                }

                // 删除中间 tensor 和 op（注意迭代器失效问题）
                tensors.erase(std::remove(tensors.begin(), tensors.end(), out), tensors.end());
                tensors.erase(std::remove(tensors.begin(), tensors.end(), outTensor), tensors.end());

                // 删除当前 op 和 next op
                it = ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
                ops.erase(std::remove(ops.begin(), ops.end(), next), ops.end());

                // 重置迭代器到开始，因为图结构已经改变
                it = ops.begin();
                changed = true;
                break;
            }

            if (changed)
                continue;

            // ===============================
            // Pass 2: Transpose 融合进 MatMul
            // ===============================
            // 使用索引循环避免迭代器失效
            for (size_t idx = 0; idx < ops.size(); ++idx)
            {
                auto op = ops[idx];
                if (!isMatMul(op))
                    continue;

                auto inputs = op->getInputs();
                for (int i = 0; i < (int)inputs.size(); ++i)
                {
                    auto t = inputs[i];
                    auto src = t->getSource();
                    if (!src || !isTranspose(src))
                        continue;

                    if (!isSwapLast2Dims(src))
                        continue;

                    // 融合到 MatMul 属性
                    auto multOp = std::dynamic_pointer_cast<MatmulObj>(op);
                    if (!multOp) continue;
                    
                    if (i == 0)
                    {
                        bool transA = multOp->getTransA();
                        multOp->setTransA(!transA);
                    }
                    else if (i == 1)
                    {
                        bool transB = multOp->getTransB();
                        multOp->setTransB(!transB);
                    }

                    // 重连输入
                    auto realInput = src->getInputs()[0];
                    op->replaceInput(t, realInput);
                    realInput->addTarget(op);

                    // 删除 transpose 操作符和中间 tensor
                    tensors.erase(std::remove(tensors.begin(), tensors.end(), t), tensors.end());
                    ops.erase(std::remove(ops.begin(), ops.end(), src), ops.end());

                    changed = true;
                    break;
                }

                if (changed)
                    break;
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        IT_ASSERT(topo_sort() == true);

        // 每个 tensor 的最后使用位置
        std::unordered_map<Tensor, int> lastUse;

        for (int i = 0; i < (int)ops.size(); ++i)
        {
            auto &op = ops[i];
            for (auto &t : op->getInputs())
            {
                lastUse[t] = i;
            }
            for (auto &t : op->getOutputs())
            {
                lastUse[t] = i;
            }
        }

        // tensor -> offset
        std::unordered_map<Tensor, size_t> addrMap;

        for (int i = 0; i < (int)ops.size(); ++i)
        {
            auto &op = ops[i];

            // 给输出 tensor 分配内存
            for (auto &t : op->getOutputs())
            {
                size_t size = t->getBytes();
                size_t addr = allocator.alloc(size);
                addrMap[t] = addr;
            }

            // 释放不再使用的 tensor
            for (auto &t : op->getInputs())
            {
                if (lastUse[t] == i)
                {
                    allocator.free(addrMap[t], t->getBytes());
                }
            }
        }

        // 真正申请一块大内存
        void *base = allocator.getPtr();

        // 绑定 tensor 数据指针
        for (auto &t : tensors)
        {
            if (addrMap.count(t))
            {
                char *ptr = static_cast<char *>(base) + addrMap[t];
                auto blob = make_ref<BlobObj>(runtime, ptr);
                t->setDataBlob(blob);
            }
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini