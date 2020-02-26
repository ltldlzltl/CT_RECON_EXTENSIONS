/*
 * @Description: classes for 2-D parallel forward projection and their gradient
 * @Author: Tianling Lyu
 * @Date: 2019-11-04 14:56:57
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-09 11:47:52
 */

#ifndef _CT_RECON_EXT_FP_PAR_2D_H_
#define _CT_RECON_EXT_FP_PAR_2D_H_

//#if USE_CUDA
#include <cuda_runtime.h>
//#endif
#include <cstdio>

namespace ct_recon
{

struct ParallelProjection2DParam
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
    double fov; // image FoV

    // Ctor
    ParallelProjection2DParam() {}
    ParallelProjection2DParam(unsigned int ns, 
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
}; // struct ParallelProjection2DParam

/*****************************************************************************/
/*                             Abstract classes                              */
/*****************************************************************************/

// abstract class for 2-D parallel projection preparation
class ParallelProjection2DPrepare
{
public:
	// Ctor and Dtor
	ParallelProjection2DPrepare(const ParallelProjection2DParam& param)
		: param_(param)
	{}
	virtual ~ParallelProjection2DPrepare() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(double* sincostbl, double* buffer1,
		int* buffer2) const = 0;
//#if USE_CUDA
	virtual bool calculate_on_gpu(double* sincostbl, double* buffer1,
		int* buffer2, cudaStream_t) const = 0;
//#endif

protected:
	ParallelProjection2DParam param_;
}; // class ParallelProjection2DPrepare

// abstract class for 2-D parallel projection
template <typename T>
class ParallelProjection2D
{
public:
	// Ctor and Dtor
	ParallelProjection2D(const ParallelProjection2DParam& param)
		: param_(param)
	{}
	virtual ~ParallelProjection2D() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(const T* img, T* proj,
		const double* sincostbl, const double* buffer1,
		const int* buffer2) const = 0;
//#if USE_CUDA
	virtual bool calculate_on_gpu(const T* img, T* proj,
		const double* sincostbl, const double* buffer1,
		const int* buffer2, cudaStream_t) const = 0;
//#endif

protected:
	ParallelProjection2DParam param_;
}; // class ParallelProjection2D

// abstract class for 2-D parallel projection gradient prepare
class ParallelProjection2DGradPrepare
{
public:
    // Ctor and Dtor
    ParallelProjection2DGradPrepare(const ParallelProjection2DParam& param)
        : param_(param)
    {}
    virtual ~ParallelProjection2DGradPrepare() {}
    // utility functions
    virtual bool calculate_on_cpu(double* buffer1, double* buffer2, 
        int* buffer3) const = 0;
//#if USE_CUDA
    virtual bool calculate_on_gpu(double* buffer1, double* buffer2, 
        int* buffer3, cudaStream_t) const = 0;
//#endif

protected:
    ParallelProjection2DParam param_;
}; // class ParallelProjection2DGradPrepare

// abstract class for 2-D parallel projection gradient
template <typename T>
class ParallelProjection2DGrad
{
public:
	// Ctor and Dtor
	ParallelProjection2DGrad(const ParallelProjection2DParam& param)
		: param_(param)
	{}
	virtual ~ParallelProjection2DGrad() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(const T* proj, T* img,
		const double* sincostbl, const double* buffer1,
		const int* buffer2) const = 0;
//#if USE_CUDA
	virtual bool calculate_on_gpu(const T* proj, T* img,
		const double* sincostbl, const double* buffer1,
		const int* buffer2, cudaStream_t) const = 0;
//#endif

protected:
	ParallelProjection2DParam param_;
}; // class ParallelProjection2DGrad


/*****************************************************************************/
/*                Projection Classes using Ray-casting                       */
/*****************************************************************************/
class ParallelProjection2DRayCastingPrepare: public ParallelProjection2DPrepare
{
public:
    // Ctor and Dtor
    ParallelProjection2DRayCastingPrepare(const ParallelProjection2DParam& param, 
                                const double step_size=0.125)
        : ParallelProjection2DPrepare(param), step_size_(step_size)
    {}
    ~ParallelProjection2DRayCastingPrepare() {}
    // utility functions
    bool calculate_on_cpu(double* sincostbl, double* begins, 
        int* nsteps) const override;
//#if USE_CUDA
    bool calculate_on_gpu(double* sincostbl, double* begins, 
        int* nsteps, cudaStream_t) const override;
//#endif

private:
    // ray forward step size proportion, 
    // actual step size is step_size_*dx
    // assuming dx==dy
    double step_size_; 
}; // class ParallelProjection2DRayCastingPrepare


template <typename T>
class ParallelProjection2DRayCasting: public ParallelProjection2D<T>
{
public:
    // Ctor and Dtor
    ParallelProjection2DRayCasting(const ParallelProjection2DParam& param, 
								   const double step_size=0.125)
		: ParallelProjection2D<T>(param), step_size_(step_size)
    {}
    ~ParallelProjection2DRayCasting()
    {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* proj, 
        const double* sincostbl, const double* begins,
		const int* nsteps) const override;
//#if USE_CUDA
    bool calculate_on_gpu(const T* img, T* proj, 
        const double* sincostbl, const double* begins,
		const int* nsteps, cudaStream_t) const override;
//#endif

private:
	double step_size_; // ray forward step size proportion
}; // class ParallelProjection2D


/*****************************************************************************/
/*                Projection Classes using ray-driven                        */
/*****************************************************************************/
class ParallelProjection2DRayDrivenPrepare: public ParallelProjection2DPrepare
{
public:
    // Ctor and Dtor
    ParallelProjection2DRayDrivenPrepare(const ParallelProjection2DParam& param)
        : ParallelProjection2DPrepare(param)
    {}
    ~ParallelProjection2DRayDrivenPrepare() {}
    // utility functions
    bool calculate_on_cpu(double* sincostbl, double* beginoffset, 
        int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(double* sincostbl, double* beginoffset, 
        int* usex, cudaStream_t) const override;
//#endif
}; // class ParallelProjection2DRayDrivenPrepare

template <typename T>
class ParallelProjection2DRayDriven: public ParallelProjection2D<T>
{
public:
// Ctor and Dtor
    ParallelProjection2DRayDriven(const ParallelProjection2DParam& param)
        : ParallelProjection2D<T>(param)
    {}
    ~ParallelProjection2DRayDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* proj, 
        const double* sincostbl, const double* beginoffset, 
        const int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(const T* img, T* proj, 
        const double* sincostbl, const double* beginoffset, 
        const int* usex, cudaStream_t) const override;
//#endif
}; // class ParallelProjection2DRayDriven

class ParallelProjection2DRayDrivenGradPrep: public ParallelProjection2DGradPrepare
{
public:
    // Ctor and Dtor
    ParallelProjection2DRayDrivenGradPrep(const ParallelProjection2DParam& param)
        : ParallelProjection2DGradPrepare(param)
    {}
    ~ParallelProjection2DRayDrivenGradPrep() {}
    // utility functions
    bool calculate_on_cpu(double* weights, double* pos, int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(double* weights, double* pos, int* usex, 
        cudaStream_t stream) const override;
//#endif
}; // ParallelProjection2DRayDrivenGradPrep

template <typename T>
class ParallelProjection2DRayDrivenGrad: public ParallelProjection2DGrad<T>
{
public:
    // Ctor and Dtor
    ParallelProjection2DRayDrivenGrad(const ParallelProjection2DParam& param)
        : ParallelProjection2DGrad<T>(param)
    {}
    ~ParallelProjection2DRayDrivenGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* weights, 
        const double* pos, const int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(const T* proj, T* img, const double* weights, 
        const double* pos, const int* usex, cudaStream_t stream) const override;
//#endif
}; // class ParallelProjection2DRayDrivenGrad

/*****************************************************************************/
/*        Projection Classes using simplified distance-driven                */
/*****************************************************************************/
class ParallelProjection2DDisDrivenPrep: public ParallelProjection2DPrepare
{
public:
    // Ctor and Dtor
    ParallelProjection2DDisDrivenPrep(const ParallelProjection2DParam& param)
        : ParallelProjection2DPrepare(param)
    {}
    ~ParallelProjection2DDisDrivenPrep() {}
    // utility functions
    bool calculate_on_cpu(double* sincostbl, double* beginoffset, 
        int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(double* sincostbl, double* beginoffset, 
        int* usex, cudaStream_t) const override;
//#endif
}; // class ParallelProjection2DDisDrivenPrepare

template <typename T>
class ParallelProjection2DDisDriven: public ParallelProjection2D<T>
{
public:
// Ctor and Dtor
    ParallelProjection2DDisDriven(const ParallelProjection2DParam& param)
        : ParallelProjection2D<T>(param)
    {}
    ~ParallelProjection2DDisDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* proj, 
        const double* sincostbl, const double* beginoffset, 
        const int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(const T* img, T* proj, 
        const double* sincostbl, const double* beginoffset, 
        const int* usex, cudaStream_t) const override;
//#endif
}; // class ParallelProjection2DDisDriven

class ParallelProjection2DDisDrivenGradPrep: public ParallelProjection2DGradPrepare
{
public:
    // Ctor and Dtor
    ParallelProjection2DDisDrivenGradPrep(const ParallelProjection2DParam& param)
        : ParallelProjection2DGradPrepare(param)
    {}
    ~ParallelProjection2DDisDrivenGradPrep() {}
    // utility functions
    bool calculate_on_cpu(double* weights, double* pos, int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(double* weights, double* pos, int* usex, 
        cudaStream_t stream) const override;
//#endif
}; // ParallelProjection2DDisDrivenGradPrep

template <typename T>
class ParallelProjection2DDisDrivenGrad: public ParallelProjection2DGrad<T>
{
public:
    // Ctor and Dtor
    ParallelProjection2DDisDrivenGrad(const ParallelProjection2DParam& param)
        : ParallelProjection2DGrad<T>(param)
    {}
    ~ParallelProjection2DDisDrivenGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* weights, 
        const double* pos, const int* usex) const override;
//#if USE_CUDA
    bool calculate_on_gpu(const T* proj, T* img, const double* weights, 
        const double* pos, const int* usex, cudaStream_t stream) const override;
//#endif
}; // class ParallelProjection2DDisDrivenGrad

} // namespace ct_recon

#endif
