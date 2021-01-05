/*
 * @Description: classes for 2-D fan backprojection algorithms and their gradients
 * @Author: Tianling Lyu
 * @Date: 2020-12-13 19:33:04
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-05 10:36:36
 */

#ifndef _CT_RECON_EXT_BP_FAN_2D_H_
#define _CT_RECON_EXT_BP_FAN_2D_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdio>

namespace ct_recon
{

struct FanBackprojection2DParam
{
    unsigned int ns; // number of channels
    unsigned int na; // number of views
    double ds; // angle between nearby channels, in rad
    double orbit; // delta source angle between nearby views, in rad
    double offset_s; // difference between center channel and (ns-1)/2
    double orbit_start; // source angle at view 0
    unsigned int nx; // number of pixels on image in x-axis
    unsigned int ny; // number of pixels on image in y-axis
    double dx; // pixel spacing in x-axis
    double dy; // pixel spacing in y-axis
    double offset_x; // difference between iso center and (nx-1)/2
    double offset_y; // difference between iso center and (ny-1)/2
    double dso; // distance between x-ray source and rotation center
    double dsd; // distance between x-ray source and detector center
    double fov; // image FoV

    // Ctor
    FanBackprojection2DParam() {}
    FanBackprojection2DParam(unsigned int ns, 
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
                                  double dso, 
                                  double dsd, 
                                  double fov)
        : ns(ns), na(na), ds(ds), orbit(orbit), offset_s(offset_s), 
        orbit_start(orbit_start), nx(nx), ny(ny), dx(dx), dy(dy), 
        offset_x(offset_x), offset_y(offset_y), dso(dso), dsd(dsd), fov(fov)
    {}
}; // struct FanBackprojection2DParam

/*****************************************************************************/
/*                             Abstract classes                              */
/*****************************************************************************/

// abstract class for 2-D fan backprojection preparation
class FanBackprojection2DPrepare
{
public:
	// Ctor and Dtor
	FanBackprojection2DPrepare(const FanBackprojection2DParam& param)
		: param_(param)
	{}
	virtual ~FanBackprojection2DPrepare() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(double* buffer1, double* buffer2,
		double* buffer3) const = 0;
#ifdef USE_CUDA
	virtual bool calculate_on_gpu(double* buffer1, double* buffer2,
		double* buffer3, cudaStream_t) const = 0;
#endif

protected:
	FanBackprojection2DParam param_;
}; // class FanProjection2DPrepare

// abstract class for 2-D fan backprojection
template <typename T>
class FanBackprojection2D
{
public:
    // Ctor and Dtor
    FanBackprojection2D(const FanBackprojection2DParam& param)
        : param_(param)
    {}
    virtual ~FanBackprojection2D() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const double*) const = 0;
#ifdef USE_CUDA
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const double*, cudaStream_t) const = 0;
#endif

protected:
    FanBackprojection2DParam param_;
}; // class FanBackprojection2D

// abstract class for 2-D Fan backprojection gradient preparation
class FanBackprojection2DGradPrep
{
public:
	// Ctor and Dtor
	FanBackprojection2DGradPrep(const FanBackprojection2DParam& param)
		: param_(param)
	{}
	virtual ~FanBackprojection2DGradPrep() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(double* buffer1, double* buffer2,
		double* buffer3) const = 0;
#ifdef USE_CUDA
	virtual bool calculate_on_gpu(double* buffer1, double* buffer2,
		double* buffer3, cudaStream_t) const = 0;
#endif

protected:
	FanBackprojection2DParam param_;
}; // class FanProjection2DGradPrep

// abstract class for 2-D Fan backprojection gradient
template <typename T>
class FanBackprojection2DGrad
{
public:
    // Ctor and Dtor
    FanBackprojection2DGrad(const FanBackprojection2DParam& param)
        : param_(param)
    {}
    virtual ~FanBackprojection2DGrad() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const double*) const = 0;
#ifdef USE_CUDA
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const double*, cudaStream_t) const = 0;
#endif

protected:
    FanBackprojection2DParam param_;
}; // class FanBackprojection2DGrad


/*****************************************************************************/
/*                      pixel-driven backprojection                          */
/*****************************************************************************/
// pre-calculate as much as possible
class FanBackprojection2DPixDrivenPrep: 
    public FanBackprojection2DPrepare
{
public:
	// Ctor and Dtor
	FanBackprojection2DPixDrivenPrep(const FanBackprojection2DParam& param)
		: FanBackprojection2DPrepare(param)
	{}
	virtual ~FanBackprojection2DPixDrivenPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(double* xpos, double* ypos,
		double* sincostbl) const override;
#ifdef USE_CUDA
	bool calculate_on_gpu(double* xpos, double* ypos,
		double* sincostbl, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DPixDrivenPrep

template <typename T>
class FanBackprojection2DPixDriven: public FanBackprojection2D<T>
{
public:
    // Ctor and Dtor
    FanBackprojection2DPixDriven(const FanBackprojection2DParam& param)
        : FanBackprojection2D<T>(param)
    {}
    ~FanBackprojection2DPixDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* xpos, 
        const double* ypos, const double*) const override;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* proj, T* img, const double* xpos, 
        const double* ypos, const double*, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DPixDriven

class FanBackprojection2DPixDrivenGradPrep
    : public FanBackprojection2DGradPrep
{
public:
	// Ctor and Dtor
	FanBackprojection2DPixDrivenGradPrep(const FanBackprojection2DParam& param)
		: FanBackprojection2DGradPrep(param)
	{}
	~FanBackprojection2DPixDrivenGradPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(double* xpos, double* ypos,
		double* sincostbl) const override;
#ifdef USE_CUDA
	bool calculate_on_gpu(double* xpos, double* ypos,
		double* sincostbl, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DPixDrivenGradPrep

template <typename T>
class FanBackprojection2DPixDrivenGrad
    : public FanBackprojection2DGrad<T>
{
public:
    // Ctor and Dtor
    FanBackprojection2DPixDrivenGrad(const FanBackprojection2DParam& param)
        : FanBackprojection2DGrad<T>(param)
    {}
    ~FanBackprojection2DPixDrivenGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* grad, const double*, 
        const double*, const double*) const override;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* img, T* grad, const double*, 
        const double*, const double*, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DPixDrivenGrad

} // namespace ct_recon

#endif