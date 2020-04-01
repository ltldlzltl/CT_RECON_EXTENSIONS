/*
 * @Description: implementation of include/helical_interpolation.h
 * @Author: Tianling Lyu
 * @Date: 2020-02-17 14:44:32
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2020-03-26 09:24:56
 */

#include "include/helical_interpolation.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

namespace ct_recon{

// helper for 360LI
template <typename T>
bool helical_interpolation_360_(const HelicalInterpolationParam& param,
    const double z, const T* proj_in, T* proj_out)
{
    int center_view = int(std::round((z - param.couch_begin) / param.couch_mov));
    int begin = center_view - param.view_per_rot;
    int end = center_view + param.view_per_rot - 1;
    if (begin < 0 || end >= param.na)
        throw std::runtime_error("View not enough for this slice!");
    // output starts at the same view angle
    int first_view = (begin / param.view_per_rot + 1) * param.view_per_rot;
    double center_t = (param.nt - 1) / 2.0 + param.offset_t;
    double center_s = (param.ns - 1) / 2.0 + param.offset_s;
    for (int iview = 0; iview < param.view_per_rot; ++iview) {
        int cur_view = first_view + iview;
        double z_dif = z - param.couch_begin - cur_view * param.couch_mov;
        int cor_view = cur_view >= center_view ? cur_view - param.view_per_rot : cur_view + param.view_per_rot;
        double z_dif2 = z - param.couch_begin - cor_view * param.couch_mov;
        for (int is = 0; is < param.ns; ++is) {
            double s = (is - center_s) * param.ds;
            double gamma = s / param.dsd;
            // current view
            double upper_t = center_t - (z_dif + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            double bottom_t = center_t - (z_dif - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            if (bottom_t < 0)
                bottom_t = 1;
            if (upper_t >= param.nt)
                upper_t = param.nt-1;
            upper_t = upper_t < 0 ? 0 : upper_t;
            bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
            double cur_value = 0;
            int count = 0;
            for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                cur_value += proj_in[is + (it + cur_view * param.nt) * param.ns];
                ++count;
            }
            cur_value /= count; // average in thick slice

            // corresponding view
            upper_t = center_t - (z_dif2 + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            bottom_t = center_t - (z_dif2 - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            if (bottom_t < 0)
                bottom_t = 1;
            if (upper_t >= param.nt)
                upper_t = param.nt-1;
            upper_t = upper_t < 0 ? 0 : upper_t;
            bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
            double cor_value = 0;
            count = 0;
            for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                cor_value += proj_in[is + (it + cor_view * param.nt) * param.ns];
                ++count;
            }
            cor_value /= count; // average in thick slice

            // interpolation
            proj_out[is + iview * param.ns] = (z_dif * cor_value - z_dif2 * cur_value) / (z_dif - z_dif2);
        }
    }
    return true;
}

template <typename T>
bool helical_interpolation_360_2_(const HelicalInterpolationParam& param,
    const double z, const T* proj_in, T* proj_out)
{
    int center_view = int(std::round((z - param.couch_begin) / param.couch_mov));
    int begin = center_view - param.view_per_rot;
    int end = center_view + param.view_per_rot - 1;
    if (begin < 0 || end >= param.na)
        throw std::runtime_error("View not enough for this slice!");
    // output starts at the same view angle
    int first_view = (begin / param.view_per_rot + 1) * param.view_per_rot;
    double center_t = (param.nt - 1) / 2.0 + param.offset_t;
    double center_s = (param.ns - 1) / 2.0 + param.offset_s;
    for (int iview = 0; iview < param.view_per_rot; ++iview) {
        int cur_view = first_view + iview;
        double z_dif = z - param.couch_begin - cur_view * param.couch_mov;
        int cor_view = cur_view >= center_view ? cur_view - param.view_per_rot : cur_view + param.view_per_rot;
        double z_dif2 = z - param.couch_begin - cor_view * param.couch_mov;
        for (int is = 0; is < param.ns; ++is) {
            double s = (is - center_s) * param.ds;
            double gamma = s / param.dsd;
            bool use_cur = true;
            bool use_cor = true;
            double cur_value = 0;
            // current view
            double upper_t = center_t - (z_dif + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            double bottom_t = center_t - (z_dif - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            if (bottom_t < 0 || upper_t >= param.nt)
                use_cur = false;
            else {
                upper_t = upper_t < 0 ? 0 : upper_t;
                bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
                int count = 0;
                for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                    cur_value += proj_in[is + (it + cur_view * param.nt) * param.ns];
                    ++count;
                }
                cur_value /= count; // average in thick slice
            }

            // corresponding view
            upper_t = center_t - (z_dif2 + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            bottom_t = center_t - (z_dif2 - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            double cor_value = 0;
            if (bottom_t < 0 || upper_t >= param.nt)
                use_cor = false;
            else {
                upper_t = upper_t < 0 ? 0 : upper_t;
                bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
                int count = 0;
                for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                    cor_value += proj_in[is + (it + cor_view * param.nt) * param.ns];
                    ++count;
                }
                cor_value /= count; // average in thick slice
            }

            // interpolation
            if (use_cur) {
                if (use_cor)
                    proj_out[is + iview * param.ns] = (z_dif * cor_value - z_dif2 * cur_value) / (z_dif - z_dif2);
                else
                    proj_out[is + iview * param.ns] = cur_value;
            }
            else {
                if (use_cor)
                    proj_out[is + iview * param.ns] = cor_value;
                else
                    throw std::runtime_error("No row data available!");
            }
        }
    }
    return true;
}

// helper for 180LI
template <typename T>
bool helical_interpolation_180_2_(const HelicalInterpolationParam& param,
    const double z, const T* proj_in, T* proj_out)
{
    int center_view = int(std::round((z - param.couch_begin) / param.couch_mov));
    int begin = center_view - param.view_per_rot / 2;
    int end = begin + param.view_per_rot - 1;
    if (begin < 0 || end >= param.na)
        throw std::runtime_error("View not enough for this slice!"); // view not enough for this slice
    // output starts at the same view angle
    int first_view = std::ceil(begin / double(param.view_per_rot)) * param.view_per_rot;
    double center_t = (param.nt - 1) / 2.0 + param.offset_t;
    double center_s = (param.ns - 1) / 2.0 + param.offset_s;
    for (int iview = 0; iview < param.view_per_rot; ++iview) {
        int cur_view = first_view + iview;
        if (cur_view > end) cur_view -= param.view_per_rot;
        double z_dif = z - param.couch_begin - cur_view * param.couch_mov;
        for (int is = 0; is < param.ns; ++is) {
            double s = (is - center_s) * param.ds;
            double gamma = s / param.dsd;
            bool use_cur = true;
            bool use_cor = true;
            double cur_value = 0;
            // current view
            double upper_t = center_t - (z_dif + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            double bottom_t = center_t - (z_dif - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            if (bottom_t < 0 || upper_t >= param.nt)
                use_cur = false;
            else {
                upper_t = upper_t < 0 ? 0 : upper_t;
                bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
                int count = 0;
                for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                    cur_value += proj_in[is + (it + cur_view * param.nt) * param.ns];
                    ++count;
                }
                cur_value /= count; // average in thick slice
            }

            // corresponding view
            double dif_view = (M_PI - 2 * gamma) / param.orbit;
            double cor_view = cur_view - dif_view;
            cor_view = cor_view < begin ? cor_view + param.view_per_rot : cor_view;
            cor_view = cor_view > end ? cor_view - param.view_per_rot : cor_view;
            double z_dif2 = z - param.couch_begin - cor_view * param.couch_mov;
            upper_t = center_t - (z_dif2 + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            bottom_t = center_t - (z_dif2 - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            double cor_value = 0;
            if (bottom_t < 0 || upper_t >= param.nt)
                use_cor = false;
            else {
                upper_t = upper_t < 0 ? 0 : upper_t;
                bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
                // for interpolation on view direction
                int cor_view_left = static_cast<int>(std::floor(cor_view));
                int cor_view_right = static_cast<int>(std::ceil(cor_view));
                double u = cor_view - cor_view_left;
                // for interpolation on channel direction
                double ss_cor = center_s - s / param.ds;
                if (ss_cor < 0 || ss_cor > param.ns - 1) {
                    // corresponding channel out of detector
                    proj_out[is + iview * param.ns] = cur_value; // no interpolation
                    continue;
                }
                int is_cor_left = static_cast<int>(std::floor(ss_cor));
                int is_cor_right = static_cast<int>(std::ceil(ss_cor));
                double v = ss_cor - is_cor_left;
                double left_value = 0, right_value = 0;
                int count = 0;
                for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                    // interpolation on channel direction
                    left_value += (1 - v) * proj_in[is_cor_left + (it + cor_view_left * param.nt) * param.ns] + v * proj_in[is_cor_right + (it + cor_view_left * param.nt) * param.ns];
                    right_value += (1 - v) * proj_in[is_cor_left + (it + cor_view_right * param.nt) * param.ns] + v * proj_in[is_cor_right + (it + cor_view_right * param.nt) * param.ns];
                    ++count;
                }
                // interpolation on angle direction
                cor_value = ((1 - u) * left_value + u * right_value) / count;
            }

            // helical interpolation or anterpolation
            if (use_cur) {
                if (use_cor)
                    proj_out[is + iview * param.ns] = (z_dif * cor_value - z_dif2 * cur_value) / (z_dif - z_dif2);
                else
                    proj_out[is + iview * param.ns] = cur_value;
            }
            else {
                if (use_cor)
                    proj_out[is + iview * param.ns] = cor_value;
                else
                    throw std::runtime_error("No row data available!");
            }
        }
    }
    return true;
}

template <typename T>
bool helical_interpolation_180_(const HelicalInterpolationParam& param,
    const double z, const T* proj_in, T* proj_out)
{
    int center_view = int(std::round((z - param.couch_begin) / param.couch_mov));
    int begin = center_view - param.view_per_rot / 2;
    int end = begin + param.view_per_rot - 1;
    if (begin < 0 || end >= param.na)
        throw std::runtime_error("View not enough for this slice!"); // view not enough for this slice
    // output starts at the same view angle
    int first_view = std::ceil(begin / double(param.view_per_rot)) * param.view_per_rot;
    double center_t = (param.nt - 1) / 2.0 + param.offset_t;
    double center_s = (param.ns - 1) / 2.0 + param.offset_s;
    for (int iview = 0; iview < param.view_per_rot; ++iview) {
        int cur_view = first_view + iview;
        if (cur_view > end) cur_view -= param.view_per_rot;
        double z_dif = z - param.couch_begin - cur_view * param.couch_mov;
        for (int is = 0; is < param.ns; ++is) {
            double s = (is - center_s) * param.ds;
            double gamma = s / param.dsd;
            // current view
            double upper_t = center_t - (z_dif + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            double bottom_t = center_t - (z_dif - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            if (bottom_t < 0)
                bottom_t = 1;
            if (upper_t >= param.nt)
                upper_t = param.nt - 1;
            upper_t = upper_t < 0 ? 0 : upper_t;
            bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
            double cur_value = 0;
            int count = 0;
            for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                cur_value += proj_in[is + (it + cur_view * param.nt) * param.ns];
                ++count;
            }
            cur_value /= count; // average in thick slice

            // corresponding view
            double dif_view = (M_PI - 2 * gamma) / param.orbit;
            double cor_view = cur_view - dif_view;
            cor_view = cor_view < begin ? cor_view + param.view_per_rot : cor_view;
            cor_view = cor_view > end ? cor_view - param.view_per_rot : cor_view;
            double z_dif2 = z - param.couch_begin - cor_view * param.couch_mov;
            upper_t = center_t - (z_dif2 + param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            bottom_t = center_t - (z_dif2 - param.thickness / 2.0) / param.dso * cos(gamma) * param.dsd;
            if (bottom_t < 0)
                bottom_t = 1;
            if (upper_t >= param.nt)
                upper_t = param.nt - 1;
            upper_t = upper_t < 0 ? 0 : upper_t;
            bottom_t = bottom_t >= param.nt ? param.nt : bottom_t;
            // for interpolation on view direction
            int cor_view_left = static_cast<int>(std::floor(cor_view));
            int cor_view_right = static_cast<int>(std::ceil(cor_view));
            double u = cor_view - cor_view_left;
            // for interpolation on channel direction
            double ss_cor = center_s - s / param.ds;
            if (ss_cor < 0 || ss_cor > param.ns - 1) {
                // corresponding channel out of detector
                proj_out[is + iview * param.ns] = cur_value; // no interpolation
                continue;
            }
            int is_cor_left = static_cast<int>(std::floor(ss_cor));
            int is_cor_right = static_cast<int>(std::ceil(ss_cor));
            double v = ss_cor - is_cor_left;
            double left_value = 0, right_value = 0;
            count = 0;
            for (int it = std::floor(upper_t); it < std::ceil(bottom_t); ++it) {
                // interpolation on channel direction
                left_value += (1 - v) * proj_in[is_cor_left + (it + cor_view_left * param.nt) * param.ns] + v * proj_in[is_cor_right + (it + cor_view_left * param.nt) * param.ns];
                right_value += (1 - v) * proj_in[is_cor_left + (it + cor_view_right * param.nt) * param.ns] + v * proj_in[is_cor_right + (it + cor_view_right * param.nt) * param.ns];
                ++count;
            }
            // interpolation on angle direction
            double cor_value = ((1 - u) * left_value + u * right_value) / count;

            // helical interpolation or anterpolation
            proj_out[is + iview * param.ns] = (z_dif * cor_value - z_dif2 * cur_value) / (z_dif - z_dif2);
        }
    }
    return true;
}

template <typename T>
bool HelicalInterpolation360<T>::calculate_on_cpu(const T* proj_in, T* proj_out) const
{
    double z = this->param_.z0;
    T* out_ptr = proj_out;
    // interpolate each slice
    for (unsigned int iz = 0; iz < this->param_.nz; ++iz) {
        if (!helical_interpolation_360_2_<T>(this->param_, z, proj_in, out_ptr))
            return false;
        z += this->param_.dz;
        out_ptr += this->param_.ns*this->param_.view_per_rot;
    }
    return true;
}

template class HelicalInterpolation360<float>;
template class HelicalInterpolation360<double>;

template <typename T>
bool HelicalInterpolation180<T>::calculate_on_cpu(const T* proj_in, T* proj_out) const
{
    double z = this->param_.z0;
    T* out_ptr = proj_out;
    // interpolate each slice
    for (unsigned int iz = 0; iz < this->param_.nz; ++iz) {
        if (!helical_interpolation_180_2_<T>(this->param_, z, proj_in, out_ptr))
            return false;
        z += this->param_.dz;
        out_ptr += this->param_.ns * this->param_.view_per_rot;
    }
    return true;
}

template class HelicalInterpolation180<float>;
template class HelicalInterpolation180<double>;

} // namespace ct_recon