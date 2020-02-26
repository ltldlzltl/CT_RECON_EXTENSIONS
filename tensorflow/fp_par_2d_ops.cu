/*
 * @Description: 2-D forward projection operator for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-11-19 12:06:57
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-09 11:49:34
 */

#include "tensorflow/fp_par_2d_ops.h"

#include <vector>
#include <memory> // for unique_ptr

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"

#include "include/fp_par_2d.h"

namespace tensorflow
{

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_CPU_KERNEL(name, T, kernel)            \
    REGISTER_KERNEL_BUILDER(                                  \
        Name(name).Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        kernel<CPUDevice, T>);
#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(name, T, kernel)            \
    REGISTER_KERNEL_BUILDER(                                  \
        Name(name).Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        kernel<GPUDevice, T>);
#endif

// Register Operator
REGISTER_OP("ForwardProjectionParallel2D")
    .Input("img: T")
    .Output("proj: T")
    .Attr("T: { float, double }")
    .Attr("img_shape: list(int) >= 2")
    .Attr("img_space: list(float) >= 2")
    .Attr("img_offset: list(float) >= 2")
    .Attr("proj_shape: list(int) >= 2")
    .Attr("channel_space: float")
    .Attr("channel_offset: float")
    .Attr("orbit_start: float")
    .Attr("orbit: float")
    .Attr("fov: float")
    .Attr("method: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));

        shape_inference::ShapeHandle out_shape = input_shape;
        ::std::vector<int> proj_shape;
        TF_RETURN_IF_ERROR(context->GetAttr("proj_shape", &proj_shape));
        shape_inference::DimensionHandle new_height = context->UnknownDim();
        shape_inference::DimensionHandle new_width = context->UnknownDim();
        if (proj_shape.size() >= 2) {
            new_height = context->MakeDim(proj_shape.at(0));
            new_width = context->MakeDim(proj_shape.at(1));
        }
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 1, new_height, &out_shape));
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 2, new_width, &out_shape));
        context->set_output(0, out_shape);
        return Status::OK();
    });

REGISTER_OP("ForwardProjectionParallel2DGrad")
    .Input("proj: T")
    .Output("grad: T")
    .Attr("T: { float, double }")
    .Attr("img_shape: list(int) >= 2")
    .Attr("img_space: list(float) >= 2")
    .Attr("img_offset: list(float) >= 2")
    .Attr("proj_shape: list(int) >= 2")
    .Attr("channel_space: float")
    .Attr("channel_offset: float")
    .Attr("orbit_start: float")
    .Attr("orbit: float")
    .Attr("fov: float")
    .Attr("method: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));

        shape_inference::ShapeHandle out_shape = input_shape;
        ::std::vector<int> img_shape;
        TF_RETURN_IF_ERROR(context->GetAttr("img_shape", &img_shape));
        shape_inference::DimensionHandle new_height = context->UnknownDim();
        shape_inference::DimensionHandle new_width = context->UnknownDim();
        new_height = context->MakeDim(img_shape.at(0));
        new_width = context->MakeDim(img_shape.at(1));
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 1, new_height, &out_shape));
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 2, new_width, &out_shape));
        context->set_output(0, out_shape);
        return Status::OK();
    });

template <typename Device, typename T>
class ForwardProjectionParallel2DOp : public OpKernel {
public:
    ForwardProjectionParallel2DOp(OpKernelConstruction* context)
        : OpKernel(context), initialized_(false)
    {
        LOG(INFO) << "Ctor";
        // load and check parameters
        ::std::vector<int> img_shape;
        OP_REQUIRES_OK(context, context->GetAttr("img_shape", &img_shape));
        OP_REQUIRES(context, img_shape.size() == 2, 
                    errors::InvalidArgument("Img_shape must contain 2 values."));
        OP_REQUIRES(context, img_shape.at(0) > 0 && img_shape.at(1) > 0, 
                    errors::InvalidArgument("Img_shape must be positive integers. "));
        ::std::vector<float> img_space;
        OP_REQUIRES_OK(context, context->GetAttr("img_space", &img_space));
        OP_REQUIRES(context, img_space.size() == 2, 
                    errors::InvalidArgument("Img_shape must contain 2 values."));
        ::std::vector<float> img_offset;
        OP_REQUIRES_OK(context, context->GetAttr("img_offset", &img_offset));
        OP_REQUIRES(context, img_offset.size() == 2, 
                    errors::InvalidArgument("Img_offset must contain 2 values."));
        ::std::vector<int> proj_shape;
        OP_REQUIRES_OK(context, context->GetAttr("proj_shape", &proj_shape));
        OP_REQUIRES(context, proj_shape.size() == 2, 
                    errors::InvalidArgument("Proj_shape must contain 2 values."));
        OP_REQUIRES(context, proj_shape.at(0) > 0 && proj_shape.at(1) > 0, 
                    errors::InvalidArgument("Proj_shape must be positive integers. "));
        float channel_space, channel_offset;
        OP_REQUIRES_OK(context, context->GetAttr("channel_space", &channel_space));
        OP_REQUIRES_OK(context, context->GetAttr("channel_offset", &channel_offset));
        float orbit_start, orbit;
        OP_REQUIRES_OK(context, context->GetAttr("orbit_start", &orbit_start));
        OP_REQUIRES_OK(context, context->GetAttr("orbit", &orbit));
        float fov;
        OP_REQUIRES_OK(context, context->GetAttr("fov", &fov));
        ::std::string method;
        OP_REQUIRES_OK(context, context->GetAttr("method", &method));
        // construct private parameters and functors
        param_ = ct_recon::ParallelProjection2DParam(proj_shape[1], proj_shape[0], 
            channel_space, orbit, channel_offset, orbit_start, img_shape[1], 
            img_shape[0], img_space[1], img_space[0], img_offset[1], img_offset[0], 
            fov);
        if (method == "raycasting") {
            // use make_unique() after c++14
            proj_prep_ = std::unique_ptr<ct_recon::ParallelProjection2DRayCastingPrepare>
                (new ct_recon::ParallelProjection2DRayCastingPrepare(param_));
            projector_ = std::unique_ptr<ct_recon::ParallelProjection2DRayCasting<T>>
                (new ct_recon::ParallelProjection2DRayCasting<T>(param_));
            // allocate memory for buffers
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, 2}), &sincostbl_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, param_.ns, 2}), &buffer1_, nullptr);
            context->allocate_persistent(DT_INT32, TensorShape({param_.na, param_.ns}), &buffer2_, nullptr);
        } else if (method == "raydriven") {
            // check on sizes
            OP_REQUIRES(context, param_.nx == param_.ny,
                        errors::InvalidArgument("Image should have same width and height for \
                        raydriven projection. "));
            proj_prep_ = std::unique_ptr<ct_recon::ParallelProjection2DRayDrivenPrepare>
                (new ct_recon::ParallelProjection2DRayDrivenPrepare(param_));
            projector_ = std::unique_ptr<ct_recon::ParallelProjection2DRayDriven<T>>
                (new ct_recon::ParallelProjection2DRayDriven<T>(param_));
            // allocate memory for buffers
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, 2}), &sincostbl_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, param_.ns, 2}), &buffer1_, nullptr);
            context->allocate_persistent(DT_INT32, TensorShape({param_.na}), &buffer2_, nullptr);
        } else if (method == "disdriven") {
            // check on sizes
            OP_REQUIRES(context, param_.nx == param_.ny,
                        errors::InvalidArgument("Image should have same width and height for \
                        distance-driven projection. "));
            proj_prep_ = std::unique_ptr<ct_recon::ParallelProjection2DDisDrivenPrep>
                (new ct_recon::ParallelProjection2DDisDrivenPrep(param_));
            projector_ = std::unique_ptr<ct_recon::ParallelProjection2DDisDriven<T>>
                (new ct_recon::ParallelProjection2DDisDriven<T>(param_));
            // allocate memory for buffers
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, 2}), &sincostbl_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, param_.ns, 3}), &buffer1_, nullptr);
            context->allocate_persistent(DT_INT32, TensorShape({param_.na}), &buffer2_, nullptr);
        } else {
            context->CtxFailure(__FILE__, __LINE__,
                                errors::InvalidArgument("Unrecognised projection method. \
                    Only raycasting, raydriven and disdriven are available now."));
        }
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        OP_REQUIRES(context, input.dims() == 4, errors::InvalidArgument("Input must be 4-dimensional. "));
        const int64 in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
        const int64 in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
        const int64 in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
        const int64 in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
        OP_REQUIRES(context, in_height == static_cast<int64>(param_.ny), 
                    errors::InvalidArgument("Input height must equal to that given in img_shape. "));
        OP_REQUIRES(context, in_width == static_cast<int64>(param_.nx), 
                    errors::InvalidArgument("Input width must equal to that given in img_shape. "));
        OP_REQUIRES(context, in_channel == 1, 
                    errors::InvalidArgument("Input channel number must be 1. "));
        
        // initialize if needed
        if (!initialized_) {
            LaunchFpPar2DPrepOp<Device>()(context, sincostbl_.AccessTensor(context)->template flat<double>().data(), 
                buffer1_.AccessTensor(context)->template flat<double>().data(), 
                buffer2_.AccessTensor(context)->template flat<int>().data(), proj_prep_.get());
            initialized_ = true;
        }

        // calculate results
        TensorShape out_shape({in_batch, param_.na, param_.ns, 1});
        Tensor* output = nullptr;
        // allocate result tensor
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
        LaunchFpPar2DOp<Device, T>()(context, input.template flat<T>().data(), 
            output->template flat<T>().data(), 
            sincostbl_.AccessTensor(context)->template flat<double>().data(), 
            buffer1_.AccessTensor(context)->template flat<double>().data(), 
            buffer2_.AccessTensor(context)->template flat<int>().data(), 
            projector_.get(), in_batch, param_.nx*param_.ny, param_.ns*param_.na);
        
        return;
    }

private:
    // parameters and functors
    ct_recon::ParallelProjection2DParam param_;
    std::unique_ptr<ct_recon::ParallelProjection2DPrepare> proj_prep_;
    std::unique_ptr<ct_recon::ParallelProjection2D<T>> projector_;
    bool initialized_;

    // tensor buffers
    PersistentTensor sincostbl_;
    PersistentTensor buffer1_;
    PersistentTensor buffer2_;
}; // class ForwardProjectionParallel2DOp

template <typename Device, typename T>
class ForwardProjectionParallel2DGradOp : public OpKernel {
public:
    ForwardProjectionParallel2DGradOp(OpKernelConstruction* context)
        : OpKernel(context), initialized_(false)
    {
        // load and check parameters
        ::std::vector<int> img_shape;
        OP_REQUIRES_OK(context, context->GetAttr("img_shape", &img_shape));
        OP_REQUIRES(context, img_shape.size() == 2, 
                    errors::InvalidArgument("Img_shape must contain 2 values."));
        OP_REQUIRES(context, img_shape.at(0) > 0 && img_shape.at(1) > 0, 
                    errors::InvalidArgument("Img_shape must be positive integers. "));
        ::std::vector<float> img_space;
        OP_REQUIRES_OK(context, context->GetAttr("img_space", &img_space));
        OP_REQUIRES(context, img_space.size() == 2, 
                    errors::InvalidArgument("Img_shape must contain 2 values."));
        ::std::vector<float> img_offset;
        OP_REQUIRES_OK(context, context->GetAttr("img_offset", &img_offset));
        OP_REQUIRES(context, img_offset.size() == 2, 
                    errors::InvalidArgument("Img_offset must contain 2 values."));
        ::std::vector<int> proj_shape;
        OP_REQUIRES_OK(context, context->GetAttr("proj_shape", &proj_shape));
        OP_REQUIRES(context, proj_shape.size() == 2, 
                    errors::InvalidArgument("Proj_shape must contain 2 values."));
        OP_REQUIRES(context, proj_shape.at(0) > 0 && proj_shape.at(1) > 0, 
                    errors::InvalidArgument("Proj_shape must be positive integers. "));
        float channel_space, channel_offset;
        OP_REQUIRES_OK(context, context->GetAttr("channel_space", &channel_space));
        OP_REQUIRES_OK(context, context->GetAttr("channel_offset", &channel_offset));
        float orbit_start, orbit;
        OP_REQUIRES_OK(context, context->GetAttr("orbit_start", &orbit_start));
        OP_REQUIRES_OK(context, context->GetAttr("orbit", &orbit));
        float fov;
        OP_REQUIRES_OK(context, context->GetAttr("fov", &fov));
        ::std::string method;
        OP_REQUIRES_OK(context, context->GetAttr("method", &method));
        // construct private parameters and functors
        param_ = ct_recon::ParallelProjection2DParam(proj_shape[1], proj_shape[0], 
            channel_space, orbit, channel_offset, orbit_start, img_shape[1], 
            img_shape[0], img_space[1], img_space[0], img_offset[1], img_offset[0], 
            fov);
        if (method == "raycasting") {
            context->CtxFailure(__FILE__, __LINE__,
                                errors::InvalidArgument("Gradient is not yet implemented \
                        for raycasting forward projector. "));
        } else if (method == "raydriven") {
            // check on sizes
            OP_REQUIRES(context, param_.nx == param_.ny,
                        errors::InvalidArgument("Image should have same width and \
                    height for raydriven projection. "));
            grad_prep_ = std::unique_ptr<ct_recon::ParallelProjection2DRayDrivenGradPrep>
                (new ct_recon::ParallelProjection2DRayDrivenGradPrep(param_));
            gradient_ = std::unique_ptr<ct_recon::ParallelProjection2DRayDrivenGrad<T>>
                (new ct_recon::ParallelProjection2DRayDrivenGrad<T>(param_));
            // allocate memory for buffers
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na}), &buffer1_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, param_.nx, 2}), &buffer2_, nullptr);
            context->allocate_persistent(DT_INT32, TensorShape({param_.na}), &buffer3_, nullptr);
        } else if (method == "disdriven") {
            // check on sizes
            OP_REQUIRES(context, param_.nx == param_.ny,
                        errors::InvalidArgument("Image should have same width and \
                    height for raydriven projection. "));
            grad_prep_ = std::unique_ptr<ct_recon::ParallelProjection2DDisDrivenGradPrep>
                (new ct_recon::ParallelProjection2DDisDrivenGradPrep(param_));
            gradient_ = std::unique_ptr<ct_recon::ParallelProjection2DDisDrivenGrad<T>>
                (new ct_recon::ParallelProjection2DDisDrivenGrad<T>(param_));
            // allocate memory for buffers
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na}), &buffer1_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na, param_.nx, 2}), &buffer2_, nullptr);
            context->allocate_persistent(DT_INT32, TensorShape({param_.na}), &buffer3_, nullptr);
        } else {
            context->CtxFailure(__FILE__, __LINE__, 
                errors::InvalidArgument("Unrecognised projection method. \
                    Only raycasting, raydriven and disdriven are available now."));
        }
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        OP_REQUIRES(context, input.dims() == 4, errors::InvalidArgument("Input must be 4-dimensional."));
        const int64 in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
        const int64 in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
        const int64 in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
        const int64 in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
        OP_REQUIRES(context, in_height == static_cast<int64>(param_.na), 
                    errors::InvalidArgument("Input height must equal to that given in proj_shape."));
        OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                    errors::InvalidArgument("Input width must equal to that given in proj_shape."));
        OP_REQUIRES(context, in_channel == 1, 
                    errors::InvalidArgument("Input channel number must be 1."));
        
        // initialize if needed
        if (!initialized_) {
            LaunchFpPar2DGradPrepOp<Device>()(context, 
                buffer1_.AccessTensor(context)->template flat<double>().data(), 
                buffer2_.AccessTensor(context)->template flat<double>().data(), 
                buffer3_.AccessTensor(context)->template flat<int>().data(), 
                grad_prep_.get());
            initialized_ = true;
        }

        // calculate results
        TensorShape out_shape({in_batch, param_.ny, param_.nx, 1});
        Tensor* output = nullptr;
        // allocate result tensor
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
        LaunchFpPar2DGradOp<Device, T>()(context, input.template flat<T>().data(), 
            output->template flat<T>().data(), 
            buffer1_.AccessTensor(context)->template flat<double>().data(), 
            buffer2_.AccessTensor(context)->template flat<double>().data(), 
            buffer3_.AccessTensor(context)->template flat<int>().data(), 
            gradient_.get(), in_batch, param_.nx*param_.ny, param_.ns*param_.na);
        
        return;
    }

private:
    // parameters and functors
    ct_recon::ParallelProjection2DParam param_;
    std::unique_ptr<ct_recon::ParallelProjection2DGradPrepare> grad_prep_;
    std::unique_ptr<ct_recon::ParallelProjection2DGrad<T>> gradient_;
    bool initialized_;

    // tensor buffers
    PersistentTensor buffer1_;
    PersistentTensor buffer2_;
    PersistentTensor buffer3_;
}; // class ForwardProjectionParallel2DGradOp

// register class to operator
REGISTER_CPU_KERNEL("ForwardProjectionParallel2D", float, ForwardProjectionParallel2DOp)
REGISTER_CPU_KERNEL("ForwardProjectionParallel2D", double, ForwardProjectionParallel2DOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("ForwardProjectionParallel2D", float, ForwardProjectionParallel2DOp)
REGISTER_GPU_KERNEL("ForwardProjectionParallel2D", double, ForwardProjectionParallel2DOp)
#endif

REGISTER_CPU_KERNEL("ForwardProjectionParallel2DGrad", float, ForwardProjectionParallel2DGradOp)
REGISTER_CPU_KERNEL("ForwardProjectionParallel2DGrad", double, ForwardProjectionParallel2DGradOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("ForwardProjectionParallel2DGrad", float, ForwardProjectionParallel2DGradOp)
REGISTER_GPU_KERNEL("ForwardProjectionParallel2DGrad", double, ForwardProjectionParallel2DGradOp)
#endif

} // namespace tensorflow
