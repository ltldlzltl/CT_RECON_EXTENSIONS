/*
 * @Description: classes for interpolate 3-D helical sinogram into 2-D fan-beam sinograms
 * @Author: Tianling Lyu
 * @Date: 2020-02-17 11:06:18
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2020-02-17 16:02:04
 */

#ifndef _CT_RECON_EXT_HELI_INTERP_H_
#define _CT_RECON_EXT_HELI_INTERP_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdio>

namespace ct_recon
{

struct HelicalInterpolationParam
{
    unsigned int ns; // number of channels
    unsigned int nt; // number of detector rows
    unsigned int na; // number of views
    double ds; // distance between nearby channels, in mm
    double dt; // spacing between nearby detector rows, in mm
    double offset_s; // difference between center channel and (ns-1)/2
    double offset_t; // difference between center row and (nt-1)/2
    double orbit_start; // source angle at view 0
    double orbit;
    double couch_begin; // source z-position at view 0
    double couch_mov; // source z-direction movement between views
    unsigned int view_per_rot; // number of views in each rotation
    unsigned int nz; // number of image slices
    double dz; // spacing between image slices
    double z0; // position of slice 0
    double thickness; // slice thickness

    // Ctor
    HelicalInterpolationParam() {}
    HelicalInterpolationParam(unsigned int ns,
                              unsigned int nt,
                              unsigned int na,
                              double ds,
                              double dt,
                              double offset_s,
                              double offset_t,
                              double orbit_start,
                              double orbit, 
                              double couch_begin,
                              double couch_mov,
                              unsigned int view_per_rot,
                              unsigned int nz,
                              double dz,
                              double z0,
                              double thickness)
        : ns(ns), na(na), ds(ds), dt(dt), offset_s(offset_s),
          offset_t(offset_t), orbit_start(orbit_start), orbit(orbit), 
          couch_begin(couch_begin), couch_mov(couch_mov), 
          view_per_rot(view_per_rot), nz(nz), dz(dz), z0(z0), 
          thickness(thickness)
    {}
}; // struct HelicalInterpolationParam

/*****************************************************************************/
/*                             Abstract classes                              */
/*****************************************************************************/

// abstract class for helical interpolation
template <typename T>
class HelicalInterpolation
{
public:
    // Ctor and Dtor
    HelicalInterpolation(const HelicalInterpolationParam& param)
        : param_(param)
    {}
    virtual ~HelicalInterpolation() {};
    // utility functions
    virtual bool calculate_on_cpu(const T* proj_in, T* proj_out) const = 0;
    virtual bool calculate_on_gpu(const T* proj_in, T* proj_out, cudaStream_t) const = 0;

protected:
    HelicalInterpolationParam param_;
}; // class HelicalInterpolation

// class for 360LI
template <typename T>
class HelicalInterpolation360: public HelicalInterpolation<T>
{
public:
    // Ctor and Dtor
    HelicalInterpolation360(const HelicalInterpolationParam& param)
        : HelicalInterpolation<T>(param)
    {}
    ~HelicalInterpolation360() {};
    // utility functions
    bool calculate_on_cpu(const T* proj_in, T* proj_out) const override;
    bool calculate_on_gpu(const T* proj_in, T* proj_out, cudaStream_t) const override;

}; // class HelicalInterpolation360

// class for 180LI
template <typename T>
class HelicalInterpolation180: public HelicalInterpolation<T>
{
public:
    // Ctor and Dtor
    HelicalInterpolation180(const HelicalInterpolationParam& param)
        : HelicalInterpolation<T>(param)
    {}
    ~HelicalInterpolation180() {};
    // utility functions
    bool calculate_on_cpu(const T* proj_in, T* proj_out) const override;
    bool calculate_on_gpu(const T* proj_in, T* proj_out, cudaStream_t) const override;

}; // class HelicalInterpolation180

} // namespace ct_recon

#endif