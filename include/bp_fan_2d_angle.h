/*
 * @Description: 
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 11:17:29
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 11:32:50
 */

#ifndef _CT_RECON_EXT_BP_FAN_2D_ANGLE_H_
#define _CT_RECON_EXT_BP_FAN_2D_ANGLE_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdio>

namespace ct_recon
{

struct FanBackprojection2DAngleParam
{
    unsigned int ns; // number of channels
    unsigned int na; // number of views
    double ds; // angle between nearby channels, in rad
    double offset_s; // difference between center channel and (ns-1)/2
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
    FanBackprojection2DAngleParam() {}
    FanBackprojection2DAngleParam(unsigned int ns, 
                                  unsigned int na, 
                                  double ds, 
                                  double offset_s,  
                                  unsigned int nx, 
                                  unsigned int ny, 
                                  double dx, 
                                  double dy, 
                                  double offset_x, 
                                  double offset_y, 
                                  double dso, 
                                  double dsd, 
                                  double fov)
        : ns(ns), na(na), ds(ds), offset_s(offset_s), nx(nx), ny(ny), dx(dx), 
        dy(dy), offset_x(offset_x), offset_y(offset_y), dso(dso), dsd(dsd), 
        fov(fov)
    {}
}; // struct FanBackprojection2DAngleParam

/*****************************************************************************/
/*                             Abstract classes                              */
/*****************************************************************************/

// abstract class for 2-D fan backprojection preparation
class FanBackprojection2DAnglePrepare
{
public:
	// Ctor and Dtor
	FanBackprojection2DAnglePrepare(const FanBackprojection2DAngleParam& param)
		: param_(param)
	{}
	virtual ~FanBackprojection2DAnglePrepare() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(const double* angles, double* buffer1, 
        double* buffer2, double* buffer3) const = 0;
#ifdef USE_CUDA
	virtual bool calculate_on_gpu(const double* angles, double* buffer1, double* buffer2,
		double* buffer3, cudaStream_t) const = 0;
#endif

protected:
	FanBackprojection2DAngleParam param_;
}; // class FanProjection2DAnglePrepare

// abstract class for 2-D fan backprojection
template <typename T>
class FanBackprojection2DAngle
{
public:
    // Ctor and Dtor
    FanBackprojection2DAngle(const FanBackprojection2DAngleParam& param)
        : param_(param)
    {}
    virtual ~FanBackprojection2DAngle() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const double*) const = 0;
#ifdef USE_CUDA
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const double*, cudaStream_t) const = 0;
#endif

protected:
    FanBackprojection2DAngleParam param_;
}; // class FanBackprojection2DAngle

// abstract class for 2-D Fan backprojection gradient preparation
class FanBackprojection2DAngleGradPrep
{
public:
	// Ctor and Dtor
	FanBackprojection2DAngleGradPrep(const FanBackprojection2DAngleParam& param)
		: param_(param)
	{}
	virtual ~FanBackprojection2DAngleGradPrep() {}
	// utility virtual functions
	virtual bool calculate_on_cpu(const double* angles, double* buffer1, 
        double* buffer2, double* buffer3) const = 0;
#ifdef USE_CUDA
	virtual bool calculate_on_gpu(const double* angles, double* buffer1, double* buffer2,
		double* buffer3, cudaStream_t) const = 0;
#endif

protected:
	FanBackprojection2DAngleParam param_;
}; // class FanProjection2DAngleGradPrep

// abstract class for 2-D Fan backprojection gradient
template <typename T>
class FanBackprojection2DAngleGrad
{
public:
    // Ctor and Dtor
    FanBackprojection2DAngleGrad(const FanBackprojection2DAngleParam& param)
        : param_(param)
    {}
    virtual ~FanBackprojection2DAngleGrad() {}
    // utility functions
    virtual bool calculate_on_cpu(const T* proj, T* img, const double*, 
        const double*, const double*) const = 0;
#ifdef USE_CUDA
    virtual bool calculate_on_gpu(const T* proj, T* img, const double*, 
        const double*, const double*, cudaStream_t) const = 0;
#endif

protected:
    FanBackprojection2DAngleParam param_;
}; // class FanBackprojection2DAngleGrad


/*****************************************************************************/
/*                      pixel-driven backprojection                          */
/*****************************************************************************/
// pre-calculate as much as possible
class FanBackprojection2DAnglePixDrivenPrep: 
    public FanBackprojection2DAnglePrepare
{
public:
	// Ctor and Dtor
	FanBackprojection2DAnglePixDrivenPrep(const FanBackprojection2DAngleParam& param)
		: FanBackprojection2DAnglePrepare(param)
	{}
	virtual ~FanBackprojection2DAnglePixDrivenPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(const double* angles, double* xpos, double* ypos,
		double* sincostbl) const override;
#ifdef USE_CUDA
	bool calculate_on_gpu(const double* angles, double* xpos, double* ypos,
		double* sincostbl, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DAnglePixDrivenPrep

template <typename T>
class FanBackprojection2DAnglePixDriven: public FanBackprojection2DAngle<T>
{
public:
    // Ctor and Dtor
    FanBackprojection2DAnglePixDriven(const FanBackprojection2DAngleParam& param)
        : FanBackprojection2DAngle<T>(param)
    {}
    ~FanBackprojection2DAnglePixDriven() {}
    // utility functions
    bool calculate_on_cpu(const T* proj, T* img, const double* xpos, 
        const double* ypos, const double*) const override;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* proj, T* img, const double* xpos, 
        const double* ypos, const double*, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DAnglePixDriven

class FanBackprojection2DAnglePixDrivenGradPrep
    : public FanBackprojection2DAngleGradPrep
{
public:
	// Ctor and Dtor
	FanBackprojection2DAnglePixDrivenGradPrep(const FanBackprojection2DAngleParam& param)
		: FanBackprojection2DAngleGradPrep(param)
	{}
	~FanBackprojection2DAnglePixDrivenGradPrep() {}
	// utility virtual functions
	bool calculate_on_cpu(const double* angles, double* xpos, double* ypos,
		double* sincostbl) const override;
#ifdef USE_CUDA
	bool calculate_on_gpu(const double* angles, double* xpos, double* ypos,
		double* sincostbl, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DAnglePixDrivenGradPrep

template <typename T>
class FanBackprojection2DAnglePixDrivenGrad
    : public FanBackprojection2DAngleGrad<T>
{
public:
    // Ctor and Dtor
    FanBackprojection2DAnglePixDrivenGrad(const FanBackprojection2DAngleParam& param)
        : FanBackprojection2DAngleGrad<T>(param)
    {}
    ~FanBackprojection2DAnglePixDrivenGrad() {}
    // utility functions
    bool calculate_on_cpu(const T* img, T* grad, const double*, 
        const double*, const double*) const override;
#ifdef USE_CUDA
    bool calculate_on_gpu(const T* img, T* grad, const double*, 
        const double*, const double*, cudaStream_t) const override;
#endif
}; // class FanBackprojection2DAnglePixDrivenGrad

} // namespace ct_recon

#endif