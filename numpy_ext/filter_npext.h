/*
 * @Description: classes and structs for ramp filter numpy extension
 * @Author: Tianling Lyu
 * @Date: 2021-01-10 18:51:45
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-03-11 14:39:38
 */

#ifndef _NP_EXT_FILTER_H_
#define _NP_EXT_FILTER_H_

#include "include/filter.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {
    struct RampFilterRunParam
    {
        double* in;
        double* out;

        RampFilterRunParam(double* in, double* out)
            : in(in), out(out)
        {}
    };

    class RampFilterNPExt
    {
    public:
        RampFilterNPExt(const ct_recon::FilterParam& param);
        ~RampFilterNPExt();

        bool allocate();
        bool run(const RampFilterRunParam& param);
    
    private:
        double* filter_;
#ifdef USE_CUDA
        double* in_;
        double* out_;
#endif

        ct_recon::FilterParam param_;
        bool allocated_;

        ct_recon::RampFilterPrep flt_prep_;
        ct_recon::RampFilter<double> flt_;
    }; // class RampFilterNPExt

} // namespace np_ext
#endif // #ifndef _NP_EXT_FILTER_H_