/*
 * @Description: implementation of include/bp_par_2d.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-22 20:18:45
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-24 11:09:02
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
        posy = (static_cast<double>(param_.ny - 1) - centy) * param_.dy;
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

} // namespace ct_recon