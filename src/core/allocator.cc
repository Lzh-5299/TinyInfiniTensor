#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
           // First-fit 查找空闲块
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
        {
            if (it->free && it->size >= size)
            {
                size_t addr = it->offset;

                // 如果块足够大，进行切分
                if (it->size > size)
                {
                    Block rest;
                    rest.offset = it->offset + size;
                    rest.size = it->size - size;
                    rest.free = true;

                    it->size = size;
                    blocks.insert(std::next(it), rest);
                }

                it->free = false;
                return addr;
            }
        }

        // 没有空闲块，从尾部分配
        size_t addr = this->used;

        Block block;
        block.offset = addr;
        block.size = size;
        block.free = false;

        blocks.push_back(block);

        this->used += size;
        if (this->used > this->peak)
        {
            this->peak = this->used;
        }

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
        {
            if (it->offset == addr)
            {
                IT_ASSERT(!it->free);
                it->free = true;

                // 向后合并
                auto next = std::next(it);
                if (next != blocks.end() && next->free)
                {
                    it->size += next->size;
                    blocks.erase(next);
                }

                // 向前合并
                if (it != blocks.begin())
                {
                    auto prev = std::prev(it);
                    if (prev->free)
                    {
                        prev->size += it->size;
                        blocks.erase(it);
                    }
                }

                break;
            }
        }

        // 如果尾块是 free，回收 used
        while (!blocks.empty())
        {
            auto last = std::prev(blocks.end());
            if (last->free)
            {
                this->used -= last->size;
                blocks.erase(last);
            }
            else
            {
                break;
            }
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
