/*
 * @Description: implementation of filter.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 14:43:27
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-28 15:13:44
 */

#include "include/filter.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

namespace ct_recon
{

template <typename T>
bool RampFilterPrep<T>::calculate_on_cpu(T* filter)
{
    // TODO: implement this function
    return false;
}

template class RampFilterPrep<float>;
template class RampFilterPrep<double>;

template <typename T>
bool RampFilter<T>::calculate_on_cpu(const T* in, const T* filter, T* out)
{
    const T* filter_ptr = filter + param_.ns;
    const T* in_ptr = in;
    T* out_ptr = out;
    // variables
    int is, ia, ipos, ipos2;
    double sum;
    for (ia = 0; ia < param_.na; ++ia) {
        for (is = 0; is < param_.ns; ++is) {
            sum = 0;
            for (ipos = -param_.ns; ipos <= param_.ns; ++ipos) {
                ipos2 = is - ipos;
                if (ipos2 >= 0 && ipos2 < param_.ns)
                    sum += in_ptr[ipos2] * filter[ipos];
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

} // namespace ct_recon