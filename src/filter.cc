/*
 * @Description: implementation of filter.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 14:43:27
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-02-07 16:34:32
 */

#include "include/filter.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

namespace ct_recon
{

// ramp filter for parallel and flat beam
template <typename T>
bool ramp_par(T* filter, const FilterParam& param)
{
    T* pfilter = filter + param.ns;
    T ds2_inv = 1.0 / (param.ds * param.ds);
    T pi2_inv = 1.0 / (M_PI*M_PI);
    for (int ipos = -static_cast<int>(param.ns); ipos <= static_cast<int>(param.ns); ++ipos) {
        if (ipos == 0) {
            pfilter[ipos] = 0.25 * ds2_inv;
        } else if (ipos % 2 == 0) {
            pfilter[ipos] = 0;
        } else {
            pfilter[ipos] = -pi2_inv * ds2_inv  / static_cast<T>(ipos*ipos);
        }
    }
    return true;
}

// ramp filter for fan beam
template <typename T>
bool ramp_fan(T* filter, const FilterParam& param)
{
    T* pfilter = filter + param.ns;
    T pidsd2_inv = 1.0 / (M_PI*M_PI*param.dsd*param.dsd);
    T sin_angle;
    for (int ipos = -static_cast<int>(param.ns); ipos <= static_cast<int>(param.ns); ++ipos) {
        if (ipos == 0) {
            pfilter[ipos] = 0.25 / (param.ds*param.ds);
        } else if (ipos % 2 == 0) {
            pfilter[ipos] = 0;
        } else {
            sin_angle = std::sin(static_cast<T>(ipos) * param.ds / param.dsd);
            pfilter[ipos] = -pidsd2_inv  / (sin_angle*sin_angle);
        }
    }
    return true;
}

template <typename T>
bool RampFilterPrep<T>::calculate_on_cpu(T* filter) const
{
    if (param_.type == 0 || param_.type == 2) {
        if (!ramp_par(filter, param_)) return false;
    } else if (param_.type == 1) {
        if (!ramp_fan(filter, param_)) return false;
    } else {
        return false;
    }
    return true;
}

template class RampFilterPrep<float>;
template class RampFilterPrep<double>;

template <typename T>
bool RampFilter<T>::calculate_on_cpu(const T* in, const T* filter, T* out) const
{
    const T* filter_ptr = filter + param_.ns;
    const T* in_ptr = in;
    T* out_ptr = out;
    // variables
    unsigned int is, irow;
    int ipos, ipos2;
    double sum;
    for (irow = 0; irow < param_.nrow; ++irow) {
        for (is = 0; is < param_.ns; ++is) {
            sum = 0;
            for (ipos = -static_cast<int>(param_.ns); ipos <= static_cast<int>(param_.ns); ++ipos) {
                ipos2 = is - ipos;
                if (ipos2 >= 0 && ipos2 < static_cast<int>(param_.ns))
                    sum += in_ptr[ipos2] * filter_ptr[ipos];
            }
            *out_ptr = sum * param_.ds;
            ++out_ptr;
        }
        in_ptr += param_.ns;
    }
    return true;
}

template class RampFilter<float>;
template class RampFilter<double>;

// the gradient of filtering is correlating
template <typename T>
bool RampFilterGrad<T>::calculate_on_cpu(const T* in, const T* filter, T* out) const
{
    const T* filter_ptr = filter + param_.ns;
    const T* in_ptr = in;
    T* out_ptr = out;
    // variables
    unsigned int is, irow;
    int ipos, ipos2;
    double sum;
    for (irow = 0; irow < param_.nrow; ++irow) {
        for (is = 0; is < param_.ns; ++is) {
            sum = 0;
            for (ipos = -static_cast<int>(param_.ns); ipos <= static_cast<int>(param_.ns); ++ipos) {
                // change '-' into '+', others are the same
                ipos2 = is + ipos;
                if (ipos2 >= 0 && ipos2 < static_cast<int>(param_.ns))
                    sum += in_ptr[ipos2] * filter_ptr[ipos];
            }
            *out_ptr = sum * param_.ds;
            ++out_ptr;
        }
        in_ptr += param_.ns;
    }
    return true;
}

template class RampFilterGrad<float>;
template class RampFilterGrad<double>;

} // namespace ct_recon