/*
 * @Description: implementation of include/fp_par_2d.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-04 17:02:30
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-06 15:28:58
 */

#include "include/fp_par_2d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace ct_recon {

bool ParallelProjection2DRayCastingPrepare::calculate_on_cpu(double* sincostbl, 
    double* begins, int* nsteps) const
{
    // declare variables
    unsigned int is, ia;
    double angle = param_.orbit_start, s;
    double sin_angle, cos_angle;
    double step_x, step_y;
    bool usex;
    double half_subtense_squ, half_subtense;
    double x1, y1, x2, y2;
    int num_samples;
    double* tbl_ptr = sincostbl;
    double* begin_ptr = begins;
    int* nstep_ptr = nsteps;
    // useful constants
    const double x_center = static_cast<double>(param_.nx-1) / 2.0 + param_.offset_x;
    const double y_center = static_cast<double>(param_.ny-1) / 2.0 + param_.offset_y;
    const double fov_squ = param_.fov*param_.fov;
    const double s_begin = -(static_cast<double>(param_.ns-1) / 2.0 + param_.offset_s)
        * param_.ds;
    // iterate on views
    for (ia = 0; ia < param_.na; ++ia) {
        sin_angle = sin(angle); // calculate sin value
        cos_angle = cos(angle); // calculate cos value
        tbl_ptr[0] = sin_angle;
        tbl_ptr[1] = cos_angle;
        // step length on x and y direction
        step_x = step_size_*sin_angle;
        step_y = step_size_*cos_angle;
        usex = sin_angle > (1.0/1.414); // calculate nsamples on x or y direction
        s = s_begin; // distance from current channel to center channel
        for (is = 0; is < param_.ns; ++is) {
            half_subtense_squ = fov_squ - s*s;
            if (half_subtense_squ > 0) {
                half_subtense = sqrt(half_subtense_squ);
                // intersection points
                x1 = (s*cos_angle - half_subtense*sin_angle) / param_.dx + x_center;
                y1 = -(s*sin_angle + half_subtense*cos_angle) / param_.dy + y_center;
				x2 = (s*cos_angle + half_subtense*sin_angle) / param_.dx + x_center;
				y2 = -(s*sin_angle - half_subtense*cos_angle) / param_.dy + y_center;
                if (usex) num_samples = static_cast<int>((x2-x1) / step_x);
                else num_samples = static_cast<int>((y2-y1) / step_y);
                // store results
                begin_ptr[0] = x1;
                begin_ptr[1] = y1;
                *nstep_ptr = num_samples;
            }
            else {
                // not intersected with FoV
                *nstep_ptr = 0;
                begin_ptr[0] = 0;
                begin_ptr[1] = 0;
            }
            // goto next channel
            begin_ptr += 2;
            ++nstep_ptr;
			s += param_.ds;
        }
        // goto next angle
        angle += param_.orbit;
        tbl_ptr += 2;
    }
    return true;
}

template <typename T>
bool ParallelProjection2DRayCasting<T>::calculate_on_cpu(const T* img, T* proj, 
    const double* sincostbl, const double* begins, const int* nsteps) const
{
    // we don't perform any checking on the array sizes in this funtion, 
    // please ensure that the sizes are available outside
    unsigned int is, ia;
    double cos_angle, sin_angle;
    double step_x, step_y;
    double x, y, u, v;
    int nstep;
    double sum;
    int ix1, ix2, iy1, iy2, ray_index;
    const double* tbl_ptr = sincostbl;
    const double* begin_ptr = begins;
    const int* nstep_ptr = nsteps;
    T* proj_ptr = proj;
    for (ia = 0; ia < this->param_.na; ++ia) {
        cos_angle = tbl_ptr[1];
        sin_angle = tbl_ptr[0];
        step_x = step_size_ * sin_angle;
        step_y = step_size_ * cos_angle;
        for (is = 0; is < this->param_.ns; ++is) {
            // for each channel
            nstep = *nstep_ptr;
            if (nstep > 0) {
                x = begin_ptr[0];
                y = begin_ptr[1];
                sum = 0;
                for (ray_index = 0; ray_index < nstep; ++ray_index) {
                    if (x >= 0 && x <= this->param_.nx-1 && y >= 0 && y <= this->param_.ny-1) {
                        // 2-D linear interpolation
                        ix1 = floor(x);
                        ix2 = ceil(x);
                        u = x-ix1;
                        iy1 = floor(y);
                        iy2 = ceil(y);
                        v = y-iy1;
                        sum += (1-v) * ((1-u)*img[ix1+iy1*this->param_.nx] + u*img[ix2+iy1*this->param_.nx])
                            + v * ((1-u)*img[ix1+iy2*this->param_.nx] + u*img[ix2+iy2*this->param_.nx]);
                    }
                    // forward ray
                    x += step_x;
                    y += step_y;
                }
                // write to proj
                *proj_ptr = sum * step_size_ * this->param_.dx;
            } else {
                // no intersecting points
                *proj_ptr = 0.0;
            }
            // next channel
            ++proj_ptr;
            begin_ptr += 2;
            ++nstep_ptr;
        }
        // next angle
        tbl_ptr += 2;
    }
	return true;
}

template class ParallelProjection2DRayCasting<float>;
template class ParallelProjection2DRayCasting<double>;

bool ParallelProjection2DRayDrivenPrepare::calculate_on_cpu(double* sincostbl, 
    double* beginoffset, int* usex) const
{
    // decalre variables
    unsigned int ia, is;
    double angle = param_.orbit_start;
    double sin_angle, cos_angle;
    double offset, offset2, begin;
    double* tbl_ptr = sincostbl;
    double* begin_ptr = beginoffset;
    int* usex_ptr = usex;
    // useful constants
    double centx = static_cast<double>(param_.nx-1) / 2.0 + param_.offset_x;
    double centy = static_cast<double>(param_.ny-1) / 2.0 + param_.offset_y;
    double cents = static_cast<double>(param_.ns-1) / 2.0 + param_.offset_s;
    for (ia = 0; ia < param_.na; ++ia) {
        // adjust angle into range [0, 2pi)
        while (angle < 0) angle += 2*M_PI;
        while (angle >= 2*M_PI) angle -= 2*M_PI;
        // calculate sin cos values
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        tbl_ptr[0] = sin_angle;
        tbl_ptr[1] = cos_angle;
        // decide using x or y direction
        if ((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4))
            *usex_ptr = 0;
        else *usex_ptr = 1;
        // calculate begins and offsets
        if (*usex_ptr) {
            offset = param_.dy * sin_angle / (cos_angle * param_.dx);
            offset2 = param_.ds / (cos_angle * param_.dx);
            begin = centx - centy*offset - cents*offset2;
            for (is = 0; is < param_.ns; ++is) {
                begin_ptr[0] = begin;
                begin_ptr[1] = offset;
                // next channel
                begin += offset2;
                begin_ptr += 2;
            }
        } else {
            offset = param_.dx * cos_angle / (sin_angle * param_.dy);
            offset2 = -param_.ds / (sin_angle * param_.dy);
            begin = centy - centx*offset - cents*offset2;
            for (is = 0; is < param_.ns; ++is) {
                begin_ptr[0] = begin;
                begin_ptr[1] = offset;
                // next channel
                begin += offset2;
                begin_ptr += 2;
            }
        }
        // next angle
        angle += param_.orbit;
        tbl_ptr += 2;
        ++usex_ptr;
    }
    return true;
}

template <typename T>
bool ParallelProjection2DRayDriven<T>::calculate_on_cpu(const T* img, 
    T* proj, const double* sincostbl, const double* beginoffset, 
    const int* usex) const
{
    // declare variables
    unsigned int is, ia;
    unsigned int ix1, ix2, iy1, iy2;
    double dist_weight, u, x, y, offset;
    T sum;
    T* proj_ptr = proj;
    const double* tbl_ptr = sincostbl;
    const double* begin_ptr = beginoffset;
    const int* usex_ptr = usex;
    // calculate projection
    for (ia = 0; ia < this->param_.na; ++ia) {
        if (*usex_ptr) {
            dist_weight = fabs(this->param_.dy / tbl_ptr[1]); // dy/cos(angle)
            for (is = 0; is < this->param_.ns; ++is) {
                x = begin_ptr[0];
                offset = begin_ptr[1];
                sum = 0.0;
                // accumulate on each row
                for (iy1 = 0; iy1 < this->param_.ny; ++iy1) {
                    if (x >= 0 && x <= this->param_.nx-1) {
                        // linear interpolation
                        ix1 = static_cast<unsigned int>(floor(x));
                        // we use ceil here to suit to the extreme
                        // situation that x==nx-1
                        ix2 = static_cast<unsigned int>(ceil(x));
                        u = x - ix1;
                        sum += (1-u) * img[ix1+iy1*this->param_.nx] + 
                            u * img[ix2+iy1*this->param_.nx];
                    }
                    x += offset;
                }
                *proj_ptr = sum * dist_weight;
                // next channel
                ++proj_ptr;
                begin_ptr += 2;
            }
        } else {
            dist_weight = fabs(this->param_.dx / tbl_ptr[0]); // dx/sin(angle)
            for (is = 0; is < this->param_.ns; ++is) {
                y = begin_ptr[0];
                offset = begin_ptr[1];
                sum = 0.0;
                // accumulate on each row
                for (ix1 = 0; ix1 < this->param_.nx; ++ix1) {
                    if (y >= 0 && y <= this->param_.ny-1) {
                        // linear interpolation
                        iy1 = static_cast<unsigned int>(floor(y));
                        iy2 = static_cast<unsigned int>(ceil(y));
                        u = y - iy1;
                        sum += (1-u) * img[ix1+iy1*this->param_.nx] + 
                            u * img[ix1+iy2*this->param_.nx];
                    }
                    y += offset;
                }
                *proj_ptr = sum * dist_weight;
                // next channel
                ++proj_ptr;
                begin_ptr += 2;
            }
        }
        // next angle
        ++usex_ptr;
        tbl_ptr += 2;
    }
    return true;
}

template class ParallelProjection2DRayDriven<float>;
template class ParallelProjection2DRayDriven<double>;

bool ParallelProjection2DRayDrivenGradPrep::calculate_on_cpu(double* weights, 
    double* pos, int* usex) const
{
    // declare variables
    unsigned int ia, ix, iy;
    double cos_angle, sin_angle;
    double begin, offset1, offset2;
    bool b_usex;
    double* weight_ptr = weights;
    double* pos_ptr = pos;
    int* usex_ptr = usex;
    // useful constants
    const double centx = static_cast<double>(param_.nx-1) / 2.0 + param_.offset_x;
    const double centy = static_cast<double>(param_.ny-1) / 2.0 + param_.offset_y;
    const double cents = static_cast<double>(param_.ns-1) / 2.0 + param_.offset_s;
    // calculate usex
    double angle = param_.orbit_start;
    for (ia = 0; ia < param_.na; ++ia) {
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        *usex_ptr = b_usex;
        if (b_usex) {
            *weight_ptr = fabs(param_.dy / cos_angle);
            offset1 = param_.dx * cos_angle / param_.ds;
            offset2 = -param_.dy * sin_angle / param_.ds;
            begin = cents - centx * offset1 - centy * offset2;
            for (iy = 0; iy < param_.ny; ++iy) {
                pos_ptr[0] = begin;
                pos_ptr[1] = offset1;
                begin += offset2;
                pos_ptr += 2;
            }
        } else {
            *weight_ptr = fabs(param_.dx / sin_angle);
            offset1 = -param_.dy * sin_angle / param_.ds;
            offset2 = param_.dx * cos_angle / param_.ds;
            begin = cents - centy*offset1 - centx*offset2;
            for (ix = 0; ix < param_.nx; ++ix) {
                pos_ptr[0] = begin; 
                pos_ptr[1] = offset1;
                begin += offset2;
                pos_ptr += 2;
            }
        }
        angle += param_.orbit;
        ++usex_ptr;
        ++weight_ptr;
    }
    return true;
}

template <typename T>
bool ParallelProjection2DRayDrivenGrad<T>::calculate_on_cpu(const T* proj, 
    T* img, const double* weights, const double* pos, const int* usex) const
{
    // declare variables
    unsigned int ix, iy, ia, is;
    double sum, tempsum, begin, offset, mid, left, right;
    const T* proj_ptr;
    T* img_ptr = img;
    const double* weight_ptr,  *pos_ptr;
    const int* usex_ptr;
    for (iy = 0; iy < this->param_.ny; ++iy) {
        for (ix = 0; ix < this->param_.nx; ++ix) {
            usex_ptr = usex;
            sum = 0.0;
            proj_ptr = proj;
            pos_ptr = pos;
            weight_ptr = weights;
            for (ia = 0; ia < this->param_.na; ++ia) {
                if (*usex_ptr) {
                    // calculate corresponding ray range
                    begin = pos_ptr[iy<<1]; // pos[2*(iy+ia*ny)]
                    offset = pos_ptr[(iy<<1) | 1]; // pos[2*(iy+ia*ny)+1]
                    mid = begin + offset * ix; // channel at current point
                    left = ix==0 ? mid : mid-offset;
                    right = ix==this->param_.nx-1 ? mid : mid+offset;
                } else {
                    // calculate corresponding ray range
                    begin = pos[(ia*this->param_.nx + ix)<<1]; // 2*(ix+ia*nx)
                    offset = pos[((ia*this->param_.nx + ix)<<1) | 1]; // 2*(ix+ia*nx)+1
                    mid = begin + offset * iy; // channel at current point
                    left = iy==0 ? mid : mid-offset;
                    right = iy==this->param_.ny-1 ? mid : mid+offset;
                }
                if (left > right) std::swap(left, right); // make sure left <= right
                // accumulate values within the range
                tempsum = 0.0;
                for (is = ceil(left); is <= right; ++is) {
                    tempsum += (1-fabs((mid-static_cast<double>(is)) / offset))
                        * proj_ptr[is];
                }
                sum += tempsum * (*weight_ptr);
                // next angle
                ++usex_ptr;
                ++weight_ptr;
                proj_ptr += this->param_.ns;
                pos_ptr += 2*this->param_.nx;
            }
            *img_ptr = sum;
            ++img_ptr;
        }
    }
    return true;
}

template class ParallelProjection2DRayDrivenGrad<float>;
template class ParallelProjection2DRayDrivenGrad<double>;

bool ParallelProjection2DDisDrivenPrep::calculate_on_cpu(double* sincostbl, 
    double* beginoffset, int* usex) const
{
    // decalre variables
    unsigned int ia, is;
    double angle = param_.orbit_start;
    double sin_angle, cos_angle;
    double offset, offset2, begin;
    double* tbl_ptr = sincostbl;
    double* begin_ptr = beginoffset;
    int* usex_ptr = usex;
    // useful constants
    double centx = static_cast<double>(param_.nx-1) / 2.0 + param_.offset_x;
    double centy = static_cast<double>(param_.ny-1) / 2.0 + param_.offset_y;
    double cents = static_cast<double>(param_.ns-1) / 2.0 + param_.offset_s;
    for (ia = 0; ia < param_.na; ++ia) {
        // adjust angle into range [0, 2pi)
        while (angle < 0) angle += 2*M_PI;
        while (angle >= 2*M_PI) angle -= 2*M_PI;
        // calculate sin cos values
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        tbl_ptr[0] = sin_angle;
        tbl_ptr[1] = cos_angle;
        // decide using x or y direction
        if ((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4))
            *usex_ptr = 0;
        else *usex_ptr = 1;
        // calculate begins and offsets
        if (*usex_ptr) {
            offset = param_.dy * sin_angle / (cos_angle * param_.dx);
            offset2 = param_.ds / (cos_angle * param_.dx);
            begin = centx - centy*offset - cents*offset2;
            for (is = 0; is < param_.ns; ++is) {
                begin_ptr[0] = begin;
                begin_ptr[1] = offset;
                begin_ptr[2] = offset2 / 2;
                // next channel
                begin += offset2;
                begin_ptr += 3;
            }
        } else {
            offset = param_.dx * cos_angle / (sin_angle * param_.dy);
            offset2 = -param_.ds / (sin_angle * param_.dy);
            begin = centy - centx*offset - cents*offset2;
            for (is = 0; is < param_.ns; ++is) {
                begin_ptr[0] = begin;
                begin_ptr[1] = offset;
                begin_ptr[2] = offset2 / 2;
                // next channel
                begin += offset2;
                begin_ptr += 3;
            }
        }
        // next angle
        angle += param_.orbit;
        tbl_ptr += 2;
        ++usex_ptr;
    }
    return true;
}

template <typename T>
bool ParallelProjection2DDisDriven<T>::calculate_on_cpu(const T* img, 
    T* proj, const double* sincostbl, const double* beginoffset, 
    const int* usex) const
{
    // declare variables
    unsigned int is, ia;
    unsigned int ix, iy;
    double dist_weight, x, y, offset, left, right, length;
    T sum, tsum, lsum;
    T* proj_ptr = proj;
    const double* tbl_ptr = sincostbl;
    const double* begin_ptr = beginoffset;
    const int* usex_ptr = usex;
    // calculate projection
    for (ia = 0; ia < this->param_.na; ++ia) {
        if (*usex_ptr) {
            dist_weight = fabs(this->param_.dy / tbl_ptr[1]); // dy/cos(angle)
            for (is = 0; is < this->param_.ns; ++is) {
                x = begin_ptr[0];
                offset = begin_ptr[1];
                left = x - begin_ptr[2];
                right = x + begin_ptr[2];
                if (left > right) std::swap(left, right);
                sum = 0.0;
                // accumulate on each row
                for (iy = 0; iy < this->param_.ny; ++iy) {
                    tsum = 0;
                    lsum = 0;
                    for (ix = floor(left); ix < ceil(right); ++ix) {
                        if (ix < 0 || ix >= this->param_.nx-1) continue;
                        length = std::min(double(ix+1), right)
                             - std::max(double(ix), left);
                        tsum += length * (img[ix+iy*this->param_.nx]
                            + img[ix+1+iy*this->param_.nx]) / 2;
                        lsum += length;
                    }
                    if (lsum > 0)
                        sum += tsum / lsum;
                    left += offset;
                    right += offset;
                }
                *proj_ptr = sum * dist_weight;
                // next channel
                ++proj_ptr;
                begin_ptr += 3;
            }
        } else {
            dist_weight = fabs(this->param_.dx / tbl_ptr[0]); // dx/sin(angle)
            for (is = 0; is < this->param_.ns; ++is) {
                y = begin_ptr[0];
                offset = begin_ptr[1];
                left = y - begin_ptr[2];
                right = y + begin_ptr[2];
                if (left > right) std::swap(left, right);
                sum = 0.0;
                // accumulate on each row
                for (ix = 0; ix < this->param_.nx; ++ix) {
                    tsum = 0;
                    lsum = 0;
                    for (iy = floor(left); iy < ceil(right); ++iy) {
                        if (iy < 0 || iy >= this->param_.ny-1) continue;
                        length = std::min(double(iy+1), right)
                             - std::max(double(iy), left);
                        tsum += length * (img[ix+iy*this->param_.nx]
                            + img[ix+(iy+1)*this->param_.nx]) / 2;
                        lsum += length;
                    }
                    if (lsum > 0)
                        sum += tsum / lsum;
                    left += offset;
                    right += offset;
                }
                *proj_ptr = sum * dist_weight;
                // next channel
                ++proj_ptr;
                begin_ptr += 3;
            }
        }
        // next angle
        ++usex_ptr;
        tbl_ptr += 2;
    }
    return true;
}

template class ParallelProjection2DDisDriven<float>;
template class ParallelProjection2DDisDriven<double>;

bool ParallelProjection2DDisDrivenGradPrep::calculate_on_cpu(double* weights, 
    double* pos, int* usex) const
{
    // declare variables
    unsigned int ia, ix, iy;
    double cos_angle, sin_angle;
    double begin, offset1, offset2;
    bool b_usex;
    double* weight_ptr = weights;
    double* pos_ptr = pos;
    int* usex_ptr = usex;
    // useful constants
    const double centx = static_cast<double>(param_.nx-1) / 2.0 + param_.offset_x;
    const double centy = static_cast<double>(param_.ny-1) / 2.0 + param_.offset_y;
    const double cents = static_cast<double>(param_.ns-1) / 2.0 + param_.offset_s;
    // calculate usex
    double angle = param_.orbit_start;
    for (ia = 0; ia < param_.na; ++ia) {
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        *usex_ptr = b_usex;
        if (b_usex) {
            *weight_ptr = fabs(param_.dy / cos_angle);
            offset1 = param_.dx * cos_angle / param_.ds;
            offset2 = -param_.dy * sin_angle / param_.ds;
            begin = cents - centx * offset1 - centy * offset2;
            for (iy = 0; iy < param_.ny; ++iy) {
                pos_ptr[0] = begin;
                pos_ptr[1] = offset1;
                begin += offset2;
                pos_ptr += 2;
            }
        } else {
            *weight_ptr = fabs(param_.dx / sin_angle);
            offset1 = -param_.dy * sin_angle / param_.ds;
            offset2 = param_.dx * cos_angle / param_.ds;
            begin = cents - centy*offset1 - centx*offset2;
            for (ix = 0; ix < param_.nx; ++ix) {
                pos_ptr[0] = begin; 
                pos_ptr[1] = offset1;
                begin += offset2;
                pos_ptr += 2;
            }
        }
        angle += param_.orbit;
        ++usex_ptr;
        ++weight_ptr;
    }
    return true;
}

template <typename T>
bool ParallelProjection2DDisDrivenGrad<T>::calculate_on_cpu(const T* proj, 
    T* img, const double* weights, const double* pos, const int* usex) const
{
    // declare variables
    unsigned int ix, iy, ia, is;
    double sum, tempsum, begin, offset, mid, left, right, length;
    const T* proj_ptr;
    T* img_ptr = img;
    const double* weight_ptr,  *pos_ptr;
    const int* usex_ptr;
    for (iy = 0; iy < this->param_.ny; ++iy) {
        for (ix = 0; ix < this->param_.nx; ++ix) {
            usex_ptr = usex;
            sum = 0.0;
            proj_ptr = proj;
            pos_ptr = pos;
            weight_ptr = weights;
            for (ia = 0; ia < this->param_.na; ++ia) {
                if (*usex_ptr) {
                    // calculate corresponding ray range
                    begin = pos_ptr[iy<<1]; // pos[2*(iy+ia*ny)]
                    offset = pos_ptr[(iy<<1) | 1]; // pos[2*(iy+ia*ny)+1]
                    mid = begin + offset * ix; // channel at current point
                    left = ix==0 ? mid : mid-offset;
                    right = ix==this->param_.nx-1 ? mid : mid+offset;
                } else {
                    // calculate corresponding ray range
                    begin = pos[(ia*this->param_.nx + ix)<<1]; // 2*(ix+ia*nx)
                    offset = pos[((ia*this->param_.nx + ix)<<1) | 1]; // 2*(ix+ia*nx)+1
                    mid = begin + offset * iy; // channel at current point
                    left = iy==0 ? mid : mid-offset;
                    right = iy==this->param_.ny-1 ? mid : mid+offset;
                }
                if (left > right) std::swap(left, right); // make sure left <= right
                // accumulate values within the range
                tempsum = 0.0;
                for (is = ceil(left-0.5); is <= floor(right+0.5); ++is) {
                    length = std::min(static_cast<double>(is)+0.5, right)
                        - std::max(static_cast<double>(is)-0.5, left);
                    tempsum += length* proj_ptr[is] / 2;
                }
                sum += tempsum * (*weight_ptr);
                // next angle
                ++usex_ptr;
                ++weight_ptr;
                proj_ptr += this->param_.ns;
                pos_ptr += 2*this->param_.nx;
            }
            *img_ptr = sum;
            ++img_ptr;
        }
    }
    return true;
}

template class ParallelProjection2DDisDrivenGrad<float>;
template class ParallelProjection2DDisDrivenGrad<double>;

} // namespace ct_recon
