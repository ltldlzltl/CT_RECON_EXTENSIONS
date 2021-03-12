/*
 * @Description: classes for 2-D parallel backprojection algorithms and their gradients
 * @Author: Tianling Lyu
 * @Date: 2019-11-22 19:30:16
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-02-18 09:58:14
 */

#ifndef _CT_RECON_EXT_BP_PAR_2D_H_
#define _CT_RECON_EXT_BP_PAR_2D_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdio>

namespace ct_recon
{

struct ParallelBackprojection2DParam
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
    ParallelBackprojection2DParam() {}
    ParallelBackprojection2DParam(unsigned int ns, 
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
}; // struct ParallelBackprojection2DParam

/*****************************************************************************/
/*                             Abstract classes                              */
/*****************************************************************************/

// abstract class for 2-D parallel backprojection preparation
class ParallelBackprojection2DPrepare
{
public:
	// Ctor and Dtor
	ParallelBackprojection2DPrepare(const ParallelBackprojection2DParam& param)
		: param_(param)
	{}
	virtual ~ParallelBackprojection2DPrepare() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(double* buffer1, double* buffer2,
		int* buffer3) const = 0;
#ifdef USE_CUDA
	virtual bool calculate_on_gpu(double* buffer1, double* buffer2,
		int* buffer3, cudaStream_t) const = 0;
#endif

protected:
	ParallelBackprojection2DParam param_;
}; // class ParallelProjection2DPrepare

// abstract class for 2-D parallel backprojection
template <typename T>
class ParallelBackprojection2D
{
public:
    // Ctor and Dtor
    ParallelBackprojection2D(const ParallelBackprojection2DParam& param)
        : param_(param)
    {}
    virtual ~ParallelBackprojection2D() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const int*) const = 0;
#ifdef USE_CUDA
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const int*, cudaStream_t) const = 0;
#endif

protected:
    ParallelBackprojection2DParam param_;
}; // class ParallelBackprojection2D

// abstract class for 2-D parallel backprojection gradient preparation
class ParallelBackprojection2DGradPrep
{
public:
	// Ctor and Dtor
	ParallelBackprojection2DGradPrep(const ParallelBackprojection2DParam& param)
		: param_(param)
	{}
	virtual ~ParallelBackprojection2DGradPrep() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(double* buffer1, double* buffer2,
		bool* buffer3) const = 0;
#ifdef USE_CUDA
	virtual bool calculate_on_gpu(double* buffer1, double* buffer2,
		bool* buffer3, cudaStream_t) const = 0;
#endif

protected:
	ParallelBackprojection2DParam param_;
}; // class ParallelProjection2DGradPrep

// abstract class for 2-D parallel backprojection gradient
template <typename T>
class ParallelBackprojection2DGrad
{
public:
    // Ctor and Dtor
    ParallelBackprojection2DGrad(const ParallelBackprojection2DParam& param)
        : param_(param)
    {}
    virtual ~ParallelBackprojection2DGrad() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const bool*) const = 0;
#ifdef USE_CUDA
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const bool*, cudaStream_t) const = 0;
#endif

protected:
    ParallelBackprojection2DParam param_;
}; // class ParallelBackprojection2DGrad


/*****************************************************************************/
/*                      pixel-driven backprojection                          */
/*****************************************************************************/
class ParallelBackprojection2DPixDrivenPrep: 
    public ParallelBackprojection2DPrepare
{
public:
	// Ctor and Dtor
	ParallelBackprojection2DPixDrivenPrep(const ParallelBackprojection2DParam& param)
		: ParallelBackprojection2DPrepare(param)
	{}
	virtual ~ParallelBackprojection2DPixDrivenPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(double* xcos, double* ysin,
		int* buffer) const override;
#ifdef USE_CUDA
	bool calculate_on_gpu(double* xcos, double* ysin,
		int* buffer, cudaStream_t) const override;
#endif
}; // class ParallelBackprojection2DPixDrivenPrep

template <typename T>
class ParallelBackprojection2DPixDriven: public ParallelBackprojection2D<T>
{
public:
    // Ctor and Dtor
    ParallelBackprojection2DPixDriven(const ParallelBackprojection2DParam& param)
        : ParallelBackprojection2D<T>(param)
    {}
    ~ParallelBackprojection2DPixDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* xcos, 
        const double* ysin, const int*) const override;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* proj, T* img, const double* xcos, 
        const double* ysin, const int*, cudaStream_t) const override;
#endif
}; // class ParallelBackprojection2DPixDriven

class ParallelBackprojection2DPixDrivenGradPrep
    : public ParallelBackprojection2DGradPrep
{
public:
	// Ctor and Dtor
	ParallelBackprojection2DPixDrivenGradPrep(const ParallelBackprojection2DParam& param)
		: ParallelBackprojection2DGradPrep(param)
	{}
	~ParallelBackprojection2DPixDrivenGradPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(double* begins, double* offsets,
		bool* usex) const override;
#ifdef USE_CUDA
	bool calculate_on_gpu(double* begins, double* offsets,
		bool* usex, cudaStream_t) const override;
#endif
}; // class ParallelBackprojection2DPixDrivenGradPrep

template <typename T>
class ParallelBackprojection2DPixDrivenGrad
    : public ParallelBackprojection2DGrad<T>
{
public:
    // Ctor and Dtor
    ParallelBackprojection2DPixDrivenGrad(const ParallelBackprojection2DParam& param)
        : ParallelBackprojection2DGrad<T>(param)
    {}
    ~ParallelBackprojection2DPixDrivenGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* grad, const double*, 
        const double*, const bool*) const override;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* img, T* grad, const double*, 
        const double*, const bool*, cudaStream_t) const override;
#endif
}; // class ParallelBackprojection2DPixDrivenGrad

/*****************************************************************************/
/*                     distance-driven backprojection                        */
/*****************************************************************************/
class ParallelBackprojection2DDisDrivenPrep: 
    public ParallelBackprojection2DPrepare
{
public:
	// Ctor and Dtor
	ParallelBackprojection2DDisDrivenPrep(const ParallelBackprojection2DParam& param)
		: ParallelBackprojection2DPrepare(param)
	{}
	~ParallelBackprojection2DDisDrivenPrep() {}
	// utility functions
	bool calculate_on_cpu(double* xcos, double* ysin,
		int* buffer) const override { return false; };
#ifdef USE_CUDA
	bool calculate_on_gpu(double* xcos, double* ysin,
		int* buffer, cudaStream_t) const override { return false; };
#endif
}; // class ParallelBackprojection2DDisDrivenPrep

template <typename T>
class ParallelBackprojection2DDisDriven: public ParallelBackprojection2D<T>
{
public:
    // Ctor and Dtor
    ParallelBackprojection2DDisDriven(const ParallelBackprojection2DParam& param)
        : ParallelBackprojection2D<T>(param)
    {}
    ~ParallelBackprojection2DDisDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* xcos, 
        const double* ysin, const int*) const override { return false;};
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* proj, T* img, const double* xcos, 
        const double* ysin, const int*, cudaStream_t) const override { return false; };
#endif
}; // class ParallelBackprojection2DDisDriven

class ParallelBackprojection2DDisDrivenGradPrep
    : public ParallelBackprojection2DGradPrep
{
public:
	// Ctor and Dtor
	ParallelBackprojection2DDisDrivenGradPrep(const ParallelBackprojection2DParam& param)
		: ParallelBackprojection2DGradPrep(param)
	{}
	~ParallelBackprojection2DDisDrivenGradPrep() {}
	// utility functions
	bool calculate_on_cpu(double* begins, double* offsets,
		bool* usex) const override { return false; };
#ifdef USE_CUDA
	bool calculate_on_gpu(double* begins, double* offsets,
		bool* usex, cudaStream_t) const override { return false; };
#endif
}; // class ParallelBackprojection2DDisDrivenGradPrep

template <typename T>
class ParallelBackprojection2DDisDrivenGrad
    : public ParallelBackprojection2DGrad<T>
{
public:
    // Ctor and Dtor
    ParallelBackprojection2DDisDrivenGrad(const ParallelBackprojection2DParam& param)
        : ParallelBackprojection2DGrad<T>(param)
    {}
    ~ParallelBackprojection2DDisDrivenGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* grad, const double*, 
        const double*, const bool*) const override { return false; };
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* img, T* grad, const double*, 
        const double*, const bool*, cudaStream_t) const override { return false; };
#endif
}; // class ParallelBackprojection2DDisDrivenGrad

} // namespace ct_recon

#endif