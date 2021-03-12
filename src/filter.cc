/*
 * @Description: implementation of filter.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 14:43:27
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-03-11 14:30:32
 */

#include "include/filter.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

namespace ct_recon
{

// ramp filter for parallel and flat beam
bool ramp_par(double* filter, const FilterParam& param)
{
    double* pfilter = filter + param.ns;
    double ds2_inv = 1.0 / (param.ds * param.ds);
    double pi2_inv = 1.0 / (M_PI*M_PI);
    for (int ipos = -static_cast<int>(param.ns); ipos <= static_cast<int>(param.ns); ++ipos) {
        if (ipos == 0) {
            pfilter[ipos] = 0.25 * ds2_inv;
        } else if (ipos % 2 == 0) {
            pfilter[ipos] = 0;
        } else {
            pfilter[ipos] = -pi2_inv * ds2_inv  / static_cast<double>(ipos*ipos);
        }
    }
    return true;
}

// ramp filter for fan beam
bool ramp_fan(double* filter, const FilterParam& param)
{
    double* pfilter = filter + param.ns;
    double pidsd2_inv = 1.0 / (M_PI*M_PI*param.dsd*param.dsd);
    double sin_angle;
    for (int ipos = -static_cast<int>(param.ns); ipos <= static_cast<int>(param.ns); ++ipos) {
        if (ipos == 0) {
            pfilter[ipos] = 0.25 / (param.ds*param.ds);
        } else if (ipos % 2 == 0) {
            pfilter[ipos] = 0;
        } else {
            sin_angle = std::sin(static_cast<double>(ipos) * param.ds / param.dsd);
            pfilter[ipos] = -pidsd2_inv  / (sin_angle*sin_angle);
        }
    }
    return true;
}

bool RampFilterPrep::calculate_on_cpu(double* filter) const
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

template <typename T>
bool RampFilter<T>::calculate_on_cpu(const T* in, const double* filter, T* out) const
{
    const double* filter_ptr = filter + param_.ns;
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
bool RampFilterGrad<T>::calculate_on_cpu(const T* in, const double* filter, T* out) const
{
    const double* filter_ptr = filter + param_.ns;
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
                if (ipos2 >= 0 && ipos2 < static_cast<int>(param_.ns)) {
                    if (filter_ptr[ipos] > 0 || filter_ptr[ipos] < 0)
                        sum += in_ptr[ipos2] * filter_ptr[ipos];
                }
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