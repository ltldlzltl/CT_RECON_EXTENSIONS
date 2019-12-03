/*
 * @Description: classes for 2-D single-view parallel bp algorithms
 *          Each view in the input sinogram will result in an image channel.
 * @Author: Tianling Lyu
 * @Date: 2019-12-03 11:14:24
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-03 11:33:26
 */

#ifndef _CT_RECON_EXT_BP_PAR_2D_SV_H_
#define _CT_RECON_EXT_BP_PAR_2D_SV_H_

#include <cuda_runtime.h>

namespace ct_recon
{
struct ParallelSingleViewBp2DParam
{
    unsigned int ns; // number of channels
    unsigned int na; // number of views
    double ds; // distance between nearby channels, in mm
    double orbit; // delta source angle between nearby views, in rad
    double offset_s; // difference between center channel and (ns-1)/2
    double orbit_start; // source angle at view 0
    unsigned int nx; // number of pixels on image in x-axis
    unsigned int ny; // number of pixels on image in y-axis
    double dx; // pixel spacing in x-axis
    double dy; // pixel spacing in y-axis
    double offset_x; // difference between iso center and (nx-1)/2
    double offset_y; // difference between iso center and (ny-1)/2
    double fov; // image FoV, not considered

    // Ctor
    ParallelSingleViewBp2DParam() {}
    ParallelSingleViewBp2DParam(unsigned int ns, 
                                  unsigned int na, 
                                  double ds, 
                                  double orbit, 
                                  double offset_s, 
                                  double orbit_start, 
                                  unsigned int nx, 
                                  unsigned int ny, 
                                  double dx, 
                                  double dy, 
                                  double offset_x, 
                                  double offset_y, 
                                  double fov)
        : ns(ns), na(na), ds(ds), orbit(orbit), offset_s(offset_s), 
        orbit_start(orbit_start), nx(nx), ny(ny), dx(dx), dy(dy), 
        offset_x(offset_x), offset_y(offset_y), fov(fov)
    {}
}; // struct ParallelSingleViewBp2DParam

/*****************************************************************************/
/*                             Abstract classes                              */
/*****************************************************************************/

// abstract class for 2-D parallel SingleViewBp preparation
class ParallelSingleViewBp2DPrepare
{
public:
	// Ctor and Dtor
	ParallelSingleViewBp2DPrepare(const ParallelSingleViewBp2DParam& param)
		: param_(param)
	{}
	virtual ~ParallelSingleViewBp2DPrepare() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(double* buffer1, double* buffer2,
		int* buffer3) const = 0;
	virtual bool calculate_on_gpu(double* buffer1, double* buffer2,
		int* buffer3, cudaStream_t) const = 0;

protected:
	ParallelSingleViewBp2DParam param_;
}; // class ParallelProjection2DPrepare

// abstract class for 2-D parallel SingleViewBp
template <typename T>
class ParallelSingleViewBp2D
{
public:
    // Ctor and Dtor
    ParallelSingleViewBp2D(const ParallelSingleViewBp2DParam& param)
        : param_(param)
    {}
    virtual ~ParallelSingleViewBp2D() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const int*) const = 0;
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const int*, cudaStream_t) const = 0;

protected:
    ParallelSingleViewBp2DParam param_;
}; // class ParallelSingleViewBp2D

/*****************************************************************************/
/*                      pixel-driven SingleViewBp                          */
/*****************************************************************************/
class ParallelSingleViewBp2DPixDrivenPrep: 
    public ParallelSingleViewBp2DPrepare
{
public:
	// Ctor and Dtor
	ParallelSingleViewBp2DPixDrivenPrep(const ParallelSingleViewBp2DParam& param)
		: ParallelSingleViewBp2DPrepare(param)
	{}
	virtual ~ParallelSingleViewBp2DPixDrivenPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(double* xcos, double* ysin,
		int* buffer) const override;
	bool calculate_on_gpu(double* xcos, double* ysin,
		int* buffer, cudaStream_t) const override;
}; // class ParallelSingleViewBp2DPixDrivenPrep

template <typename T>
class ParallelSingleViewBp2DPixDriven: public ParallelSingleViewBp2D<T>
{
public:
    // Ctor and Dtor
    ParallelSingleViewBp2DPixDriven(const ParallelSingleViewBp2DParam& param)
        : ParallelSingleViewBp2D<T>(param)
    {}
    ~ParallelSingleViewBp2DPixDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* xcos, 
        const double* ysin, const int*) const override;
    bool calculate_on_gpu(const T* proj, T* img, const double* xcos, 
        const double* ysin, const int*, cudaStream_t) const override;
}; // class ParallelSingleViewBp2DPixDriven

} // namespace ct_recon

#endif