/*
 * @Description: abstract container for operator runners
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 18:06:32
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-11 17:47:45
 */

#ifndef _NP_EXT_COMMON_H_
#define _NP_EXT_COMMON_H_

#include <map>
#include <memory>
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {

template <typename FunctorType, typename ParamType1, typename ParamType2>
class OpContainer
{
public:
    OpContainer() 
        :functors_(), count_(0)
    {};
    ~OpContainer() {
        functors_.clear();
    }

    int create(ParamType1 param, int device) {
        auto pair = std::make_pair(++count_, std::make_shared<FunctorType>(param, device));
        functors_.insert(pair);
        return count_;
    }
    bool run(int handle, ParamType2 param) {
        auto iter = functors_.find(handle);
        if (iter == functors_.end()) return false;
        if (!iter->second->allocate()) return false;
        return iter->second->run(param);
    }
    bool erase(int handle) {
        auto iter = functors_.find(handle);
        if (iter == functors_.end()) return false;
        functors_.erase(iter);
        return true;
    }

private:
    std::map<int, std::shared_ptr<FunctorType>> functors_;
    int count_;
};

extern int device_;
#ifdef USE_CUDA
extern cudaStream_t stream_;
#endif

} // namespace np_ext

#endif