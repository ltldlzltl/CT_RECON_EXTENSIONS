/*
 * @Description: implementation of include/bp_par_2d.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-22 20:18:45
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-26 14:43:39
 */

#include "include/bp_par_2d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

namespace ct_recon
{
bool ParallelBackprojection2DPixDrivenPrep::calculate_on_cpu(double* xcos, 
    double* ysin, int* buffer) const
{
    // useful constants
    const double centx = (static_cast<double>(this->param_.nx-1)) / 2 + 
        this->param_.offset_x;
    const double centy = (static_cast<double>(this->param_.ny-1)) / 2 + 
        this->param_.offset_y;
    // variables
    unsigned int ix, iy, ia;
    double angle = param_.orbit_start;
    double sin_angle, cos_angle;
    double posx, posy;
    for (ia = 0; ia < param_.na; ++ia) {
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        // calculate x*cos_angle
        posx = -centx * param_.dx;
        for (ix = 0; ix < param_.nx; ++ix) {
            xcos[ia + ix*param_.na] = posx * cos_angle;
            posx += param_.dx;
        }
        // calculate y*sin_angle
        posy = centy * param_.dy;
        for (iy = 0; iy < param_.ny; ++iy) {
            ysin[ia + iy*param_.na] = posy * sin_angle;
            posy -= param_.dy;
        }
    }
    return true;
}

template <typename T>
bool ParallelBackprojection2DPixDriven<T>::calculate_on_cpu(const T* proj, 
    T* img, const double* xcos, const double* ysin, const int* buffer) const
{
    // variable declaration
    unsigned int ix, iy, is1, is2, ia;
    T* img_ptr = img;
    const double *xcos_ptr, *ysin_ptr = ysin;
    double posy, posx;
    double sum, tempsum, u, s;
    // useful constants
    const double cents = (static_cast<double>(this->param_.ns-1)) / 2 + 
        this->param_.offset_s;
    // start backprojection
    for (iy = 0; iy < this->param_.ny; ++iy) {
        xcos_ptr = xcos;
        for (ix = 0; ix < this->param_.nx; ++ix) {
            sum = 0.0;
            for (ia = 0; ia < this->param_.na; ++ia) {
                s = (*xcos_ptr + ysin_ptr[ia]) / this->param_.ds + cents;
                if (s >= 0 && s <= this->param_.ns-1) {
                    // linear interpolation
                    is1 = static_cast<unsigned int>(floor(s));
                    is2 = static_cast<unsigned int>(ceil(s));
                    u = s - is1;
                    sum += (1-u) * proj[is1 + ia*this->param_.ns]
                        + u * proj[is2 + ia*this->param_.ns];
                }
                // next angle
                ++xcos_ptr;
            }
            // write result to img
            *img_ptr = sum * this->param_.orbit;
            ++img_ptr;
        }
        // next y
        ysin_ptr += this->param_.na;
    }
    return true;
}

template class ParallelBackprojection2DPixDriven<float>;
template class ParallelBackprojection2DPixDriven<double>;

bool ParallelBackprojection2DPixDrivenGradPrep::calculate_on_cpu(
    double* begins, double* offsets, bool* usex) const
{
    // useful constants
    const double cents = (static_cast<double>(this->param_.ns-1)) / 2 + 
        this->param_.offset_s;
    const double centx = (static_cast<double>(this->param_.nx-1)) / 2 + 
        this->param_.offset_x;
    const double centy = (static_cast<double>(this->param_.ny-1)) / 2 + 
        this->param_.offset_y;
    // variables
    unsigned int ia, is;
    double angle = param_.orbit_start;
    double sin_angle, cos_angle;
    double offset1, offset2, begin;
    double* begin_ptr = begins, *offset_ptr = offsets;
    bool *usex_ptr = usex, b_usex;
    // begin preparation
    for (ia = 0; ia < param_.na; ++ia) {
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        // adjust angle range to [0, 2*pi)
        while (angle < 0) angle += 2*M_PI;
        while (angle >= 2*M_PI) angle -= 2*M_PI;
        b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) ||
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        *usex_ptr = b_usex;
        if (b_usex) {
            offset1 = param_.ds / (cos_angle * param_.dx);
            offset2 = param_.dy * sin_angle / (cos_angle * param_.dx);
            begin = centx - centy * offset2 - cents * offset1;
        } else {
            offset1 = -param_.ds / (sin_angle * param_.dy);
            offset2 = param_.dx * cos_angle / (sin_angle * param_.dy);
            begin = centy - centx * offset2 - cents * offset1;
        }
        *offset_ptr = offset2;
        for (is = 0; is < param_.ns; ++is) {
            *begin_ptr = begin;
            begin += offset1;
            ++begin_ptr;
        }
        // next angle
        angle += param_.orbit;
        ++usex_ptr;
        ++offset_ptr;
    }
    return true;
}

template <typename T>
bool ParallelBackprojection2DPixDrivenGrad<T>::calculate_on_cpu(const T* img, 
    T* grad, const double* begins, const double* offsets, const bool* usex) const
{
    // variables
    unsigned int is, ia, ix, iy;
    double x, y, u, offset, sum, pos, left, right, length;
    bool usex;
    T* grad_ptr;
    const double *begin_ptr, *offset_ptr;
    const bool *usex_ptr;
    // calculate gradient
    for (ia = 0; ia < this->param_.na; ++ia) {
        usex = *usex_ptr;
        offset = *offset_ptr;
        length = fabs(*(begin_ptr+1)-*begin_ptr);
        for (is = 0; is < this->param_.ns; ++is) {
            // calculate at each channel
            sum = 0.0;
            if (usex) {
                x = *begin_ptr;
                for (iy = 0; iy < this->param_.ny; ++iy) {
                    left = (is==0) ? x : *(begin_ptr-1);
                    right = (is==this->param_.ns) ? x : *(begin_ptr+1);
                    for (ix = ceil(left); ix <= right; ++ix) {
                        u = fabs(ix - x);
                        sum += (1-u/length) * img[ix + iy*this->param_.nx];
                    }
                    x += offset;
                }
            } else {
                y = *begin_ptr;
                for (ix = 0; ix < this->param_.nx; ++ix) {
                    left = (is==0) ? y : *(begin_ptr-1);
                    right = (is==this->param_.ns) ? y : *(begin_ptr+1);
                    for (iy = ceil(left); iy <= right; ++iy) {
                        u = fabs(iy - y);
                        sum += (1-u/length) * img[ix + iy*this->param_.nx];
                    }
                    y += offset;
                }
            }
            // write to gradient array
            *grad_ptr = sum * param_.orbit;
            // next channel
            ++grad_ptr;
            ++begin_ptr;
        }
        ++usex_ptr;
        ++offset_ptr;
    }
    return true;
}

template class ParallelBackprojection2DPixDrivenGrad<float>;
template class ParallelBackprojection2DPixDrivenGrad<double>;

} // namespace ct_recon