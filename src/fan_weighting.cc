/*
 * @Description: 
 * @Author: Tianling Lyu
 * @Date: 2021-01-08 17:08:56
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 09:23:59
 */

#include "include/fan_weighting.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace ct_recon
{

template <typename T>
bool FanWeighting<T>::calculate_on_cpu(const T* in, T* out) const
{
    int is, ia;
    double s, w;
    double cents = static_cast<double>(this->param_.ns-1) / 2;
    for (is = 0; is < this->param_.ns; ++is) {
        s = this->param_.ds * (static_cast<double>(is) - cents);
        if (this->param_.type == "flat"){
            w = this->param_.dso * fabs(cos(atan2(s, this->param_.dsd))) / this->param_.dsd;
        } else if (this->param_.type == "fan") {
            w = this->param_.dso * fabs(cos(s / this->param_.dsd)) / this->param_.dsd;
        } else {
            return false;
        }
        for (ia = 0; ia < this->param_.nrow; ++ia)
            out[is + ia*this->param_.ns] = w * in[is + ia*this->param_.ns];
    }
    return true;
}

template <typename T>
bool FanWeightingGrad<T>::calculate_on_cpu(const T* in, T* out) const
{
    int is, ia;
    double s, w;
    double cents = static_cast<double>(this->param_.ns-1) / 2;
    for (is = 0; is < this->param_.ns; ++is) {
        s = this->param_.ds * (static_cast<double>(is) - cents);
        if (this->param_.type == "flat"){
            w = this->param_.dso * fabs(cos(atan2(s, this->param_.dsd))) / this->param_.dsd;
        } else if (this->param_.type == "fan") {
            w = this->param_.dso * fabs(cos(s / this->param_.dsd)) / this->param_.dsd;
        } else {
            return false;
        }
        for (ia = 0; ia < this->param_.nrow; ++ia)
            out[is + ia*this->param_.ns] = in[is + ia*this->param_.ns] / w;
    }
    return true;
}

template class FanWeighting<float>;
template class FanWeighting<double>;

} // namespace ct_recon