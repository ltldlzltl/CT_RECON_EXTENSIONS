/*
 * @Description: implementation of include/bp_fan_2d.h
 * @Author: Tianling Lyu
 * @Date: 2020-12-13 20:36:29
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 10:46:48
 */

#include "include/bp_fan_2d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace ct_recon
{
bool FanBackprojection2DPixDrivenPrep::calculate_on_cpu(double* xpos, 
    double* ypos, double* sincostbl) const
{
    // useful constants
    const double centx = (static_cast<double>(this->param_.nx-1)) / 2 + 
        this->param_.offset_x;
    const double centy = (static_cast<double>(this->param_.ny-1)) / 2 + 
        this->param_.offset_y;
    // variables
    unsigned int ix, iy, ia;
    double angle = this->param_.orbit_start;
    double sin_angle, cos_angle;
    double posx = -centx * this->param_.dx, posy = centy * this->param_.dy;
    for (ia = 0; ia < this->param_.na; ++ia) {
        sincostbl[2*ia] = sin(angle);
        sincostbl[2*ia+1] = cos(angle);
        angle += param_.orbit;
    }
    for (ix = 0; ix < this->param_.nx; ++ix) {
        xpos[ix] = posx;
        posx += this->param_.dx;
    }
    for (iy = 0; iy < this->param_.ny; ++iy) {
        ypos[iy] = posy;
        posy -= this->param_.dy;
    }
    return true;
}

template <typename T>
bool FanBackprojection2DPixDriven<T>::calculate_on_cpu(const T* proj, 
    T* img, const double* xpos, const double* ypos, const double* sincostbl) const
{
    // variable declaration
    unsigned int ix, iy, is1, is2, ia;
    T* img_ptr = img;
    const double *x_ptr, *y_ptr = ypos, *a_ptr;
    double sum, u, s, d_loop, r_loop, w, gamma;
    // useful constants
    const double cents = (static_cast<double>(this->param_.ns-1)) / 2 + 
        this->param_.offset_s;
    const double factor = round(fabs(this->param_.na*this->param_.orbit) / M_PI);
    // start backprojection
    for (iy = 0; iy < this->param_.ny; ++iy) {
        x_ptr = xpos;
        for (ix = 0; ix < this->param_.nx; ++ix) {
            sum = 0.0;
            a_ptr = sincostbl;
            for (ia = 0; ia < this->param_.na; ++ia) {
                d_loop = this->param_.dso + (*x_ptr)*a_ptr[0] - (*y_ptr)*a_ptr[1];
                r_loop = (*x_ptr)*a_ptr[1] + (*y_ptr)*a_ptr[0];
                gamma = atan2(r_loop, d_loop);
                s = gamma * this->param_.dsd / this->param_.ds + cents;
                w = this->param_.dsd * this->param_.dsd / (d_loop*d_loop + r_loop*r_loop);
                if (s >= 0 && s <= this->param_.ns-1) {
                    // linear interpolation
                    is1 = static_cast<unsigned int>(floor(s));
                    is2 = static_cast<unsigned int>(ceil(s));
                    u = s - is1;
                    sum += ((1-u) * proj[is1 + ia*this->param_.ns]
                        + u * proj[is2 + ia*this->param_.ns]) * w;
                }
                // next angle
                a_ptr += 2;
            }
            // write result to img
            *img_ptr = sum * fabs(this->param_.orbit) / factor;
            ++img_ptr;
            ++x_ptr;
        }
        // next y
        ++y_ptr;
    }
    return true;
}

template class FanBackprojection2DPixDriven<float>;
template class FanBackprojection2DPixDriven<double>;

bool FanBackprojection2DPixDrivenGradPrep::calculate_on_cpu(double* xpos, 
    double* ypos, double* sincostbl) const
{
    // useful constants
    const double centx = (static_cast<double>(this->param_.nx-1)) / 2 + 
        this->param_.offset_x;
    const double centy = (static_cast<double>(this->param_.ny-1)) / 2 + 
        this->param_.offset_y;
    // variables
    unsigned int ix, iy, ia;
    double angle = this->param_.orbit_start;
    double sin_angle, cos_angle;
    double posx = -centx * this->param_.dx, posy;
    for (ia = 0; ia < this->param_.na; ++ia) {
        sincostbl[2*ia] = sin(angle);
        sincostbl[2*ia+1] = cos(angle);
        angle += param_.orbit;
    }
    for (ix = 0; ix < this->param_.nx; ++ix) {
        xpos[ix] = posx;
        posx += this->param_.dx;
    }
    for (iy = 0; iy < this->param_.ny; ++iy) {
        ypos[iy] = posy;
        posy += this->param_.dy;
    }
    return true;
}

template <typename T>
bool FanBackprojection2DPixDrivenGrad<T>::calculate_on_cpu(const T* img, 
    T* grad, const double* xpos, const double* ypos, const double* sincostbl) const
{
    // variable declaration
    unsigned int ix, iy, is1, is2, ia;
    const T* img_ptr = img;
    const double *x_ptr, *y_ptr = ypos, *a_ptr;
    double u, s, d_loop, r_loop, w;
    // useful constants
    const double cents = (static_cast<double>(this->param_.ns-1)) / 2 + 
        this->param_.offset_s;
    const double factor = round(fabs(this->param_.na*this->param_.orbit) / M_PI);
    // initialize gradient
    for (int i = 0; i < this->param_.ns*this->param_.na; ++i)
        grad[i] = 0.0;
    // start backprojection
    for (iy = 0; iy < this->param_.ny; ++iy) {
        x_ptr = xpos;
        for (ix = 0; ix < this->param_.nx; ++ix) {
            a_ptr = sincostbl;
            for (ia = 0; ia < this->param_.na; ++ia) {
                d_loop = this->param_.dso + (*x_ptr)*a_ptr[0] - (*y_ptr)*a_ptr[1];
                r_loop = (*x_ptr)*a_ptr[1] + (*y_ptr)*a_ptr[0];
                w = this->param_.dsd * this->param_.dsd / (d_loop*d_loop + r_loop*r_loop);
                s = atan2(r_loop, d_loop) * this->param_.dsd / this->param_.ds + cents;
                if (s >= 0 && s <= this->param_.ns-1) {
                    // linear interpolation
                    is1 = static_cast<unsigned int>(floor(s));
                    is2 = static_cast<unsigned int>(ceil(s));
                    u = s - is1;
                    grad[is1 + ia*this->param_.ns] += (*img_ptr) / static_cast<T>(w * (1-u));
                    grad[is2 + ia*this->param_.ns] += (*img_ptr) / static_cast<T>(w * u);
                }
                // next angle
                a_ptr += 2;
            }
            // next image position
            ++img_ptr;
            ++x_ptr;
        }
        // next y
        ++y_ptr;
    }
    // apply weight
    T weight = static_cast<T>(factor / fabs(this->param_.orbit));
    for (int i = 0; i < this->param_.ns*this->param_.na; ++i)
        grad[i] *= weight;
    return true;
}

template class FanBackprojection2DPixDrivenGrad<float>;
template class FanBackprojection2DPixDrivenGrad<double>;

} // namespace ct_recon