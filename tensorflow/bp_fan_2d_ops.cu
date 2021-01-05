/*
 * @Description: 2-D fan backprojection operator for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2021-01-05 19:02:36
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-05 20:14:27
 */

 #include "tensorflow/bp_fan_2d_ops.h"

#include <vector>
#include <memory>
#include <cstdio>

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

#include "include/bp_fan_2d.h"

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
REGISTER_OP("BackprojectionFan2D")
    .Input("proj: T")
    .Output("img: T")
    .Attr("T: { float, double }")
    .Attr("img_shape: list(int) >= 2")
    .Attr("img_space: list(float) >= 2")
    .Attr("img_offset: list(float) >= 2")
    .Attr("proj_shape: list(int) >= 2")
    .Attr("channel_space: float")
    .Attr("channel_offset: float")
    .Attr("orbit_start: float")
    .Attr("orbit: float")
    .Attr("dso: float")
    .Attr("dsd: float")
    .Attr("fov: float")
    .Attr("method: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        LOG(INFO) << "Backprojection fan 2-D shape function.";
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));

        shape_inference::ShapeHandle out_shape = input_shape;
        ::std::vector<int> img_shape;
        TF_RETURN_IF_ERROR(context->GetAttr("img_shape", &img_shape));
        shape_inference::DimensionHandle new_height = context->UnknownDim();
        shape_inference::DimensionHandle new_width = context->UnknownDim();
        if (img_shape.size() >= 2) {
            new_height = context->MakeDim(img_shape.at(0));
            new_width = context->MakeDim(img_shape.at(1));
        }
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 1, new_height, &out_shape));
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 2, new_width, &out_shape));
        context->set_output(0, out_shape);
        return Status::OK();
    });

REGISTER_OP("BackprojectionFan2DGrad")
    .Input("img: T")
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
    .Attr("dso: float")
    .Attr("dsd: float")
    .Attr("fov: float")
    .Attr("method: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        LOG(INFO) << "Backprojection fan 2-D gradient shape function.";
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));

        shape_inference::ShapeHandle out_shape = input_shape;
        ::std::vector<int> proj_shape;
        TF_RETURN_IF_ERROR(context->GetAttr("proj_shape", &proj_shape));
        shape_inference::DimensionHandle new_height = context->UnknownDim();
        shape_inference::DimensionHandle new_width = context->UnknownDim();
        new_height = context->MakeDim(proj_shape.at(0));
        new_width = context->MakeDim(proj_shape.at(1));
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 1, new_height, &out_shape));
        TF_RETURN_IF_ERROR(context->ReplaceDim(out_shape, 2, new_width, &out_shape));
        context->set_output(0, out_shape);
        return Status::OK();
    });

template <typename Device, typename T>
class BackprojectionFan2DOp : public OpKernel {
    public:
        BackprojectionFan2DOp(OpKernelConstruction* context)
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
            float dso, dsd;
            OP_REQUIRES_OK(context, context->GetAttr("dso", &dso));
            OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
            float fov;
            OP_REQUIRES_OK(context, context->GetAttr("fov", &fov));
            ::std::string method;
            OP_REQUIRES_OK(context, context->GetAttr("method", &method));
            // construct private parameters and functors
            param_ = ct_recon::FanBackprojection2DParam(proj_shape[1], proj_shape[0], 
                channel_space, orbit, channel_offset, orbit_start, img_shape[1], 
                img_shape[0], img_space[1], img_space[0], img_offset[1], img_offset[0], 
                dso, dsd, fov);
            if (method == "pixdriven") {
                // use make_unique() after c++14
                bp_prep_ = std::unique_ptr<ct_recon::FanBackprojection2DPrepare>
                    (new ct_recon::FanBackprojection2DPixDrivenPrep(param_));
                bp_ = std::unique_ptr<ct_recon::FanBackprojection2D<T>>
                    (new ct_recon::FanBackprojection2DPixDriven<T>(param_));
                // allocate memory for buffers
                context->allocate_persistent(DT_DOUBLE, TensorShape({param_.nx}), &buffer1_, nullptr);
                context->allocate_persistent(DT_DOUBLE, TensorShape({param_.ny}), &buffer2_, nullptr);
                context->allocate_persistent(DT_DOUBLE, TensorShape({2*param_.na}), &buffer3_, nullptr);
            } else {
                context->CtxFailure(__FILE__, __LINE__,
                                    errors::InvalidArgument("Unrecognised backprojection method. \
                        Only pixdriven is available for fan geometry now."));
            }
        }
    
        void Compute(OpKernelContext* context) override {
            const Tensor& input = context->input(0);
            OP_REQUIRES(context, input.dims() == 4, errors::InvalidArgument("Input must be 4-dimensional. "));
            const int64 in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            const int64 in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            const int64 in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            const int64 in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height == static_cast<int64>(param_.na), 
                        errors::InvalidArgument("Input height must equal to that given in img_shape. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in img_shape. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
            
            // initialize if needed
            if (!initialized_) {
                bool ok = LaunchBpPar2DPrepOp<Device>()(context, 
                    buffer1_.AccessTensor(context)->template flat<double>().data(), 
                    buffer2_.AccessTensor(context)->template flat<double>().data(), 
                    buffer3_.AccessTensor(context)->template flat<double>().data(), 
                    bp_prep_.get());
                initialized_ = true;
            }
    
            // calculate results
            TensorShape out_shape({in_batch, param_.ny, param_.nx, 1});
            Tensor* output = nullptr;
            // allocate result tensor
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
            bool ok = LaunchBpPar2DOp<Device, T>()(context, input.template flat<T>().data(), 
                output->template flat<T>().data(), 
                buffer1_.AccessTensor(context)->template flat<double>().data(), 
                buffer2_.AccessTensor(context)->template flat<double>().data(), 
                buffer3_.AccessTensor(context)->template flat<double>().data(), 
                bp_.get(), in_batch, param_.nx*param_.ny, param_.ns*param_.na);
            return;
        }
    
    private:
        // parameters and functors
        ct_recon::FanBackprojection2DParam param_;
        std::unique_ptr<ct_recon::FanBackprojection2DPrepare> bp_prep_;
        std::unique_ptr<ct_recon::FanBackprojection2D<T>> bp_;
        bool initialized_;
    
        // tensor buffers
        PersistentTensor buffer1_;
        PersistentTensor buffer2_;
        PersistentTensor buffer3_;
}; // class BackprojectionFan2DOp

template <typename Device, typename T>
class BackprojectionFan2DGradOp : public OpKernel {
public:
    BackprojectionFan2DGradOp(OpKernelConstruction* context)
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
        float dso, dsd;
        OP_REQUIRES_OK(context, context->GetAttr("dso", &dso));
        OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
        float fov;
        OP_REQUIRES_OK(context, context->GetAttr("fov", &fov));
        ::std::string method;
        OP_REQUIRES_OK(context, context->GetAttr("method", &method));
        // construct private parameters and functors
        param_ = ct_recon::FanBackprojection2DParam(proj_shape[1], proj_shape[0], 
            channel_space, orbit, channel_offset, orbit_start, img_shape[1], 
            img_shape[0], img_space[1], img_space[0], img_offset[1], img_offset[0], 
            fov);
        if (method == "pixdriven") {
            grad_prep_ = std::unique_ptr<ct_recon::FanBackprojection2DGradPrep>
                (new ct_recon::FanBackprojection2DPixDrivenGradPrep(param_));
            gradient_ = std::unique_ptr<ct_recon::FanBackprojection2DGrad<T>>
                (new ct_recon::FanBackprojection2DPixDrivenGrad<T>(param_));
            // allocate memory for buffers
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.nx}), &buffer1_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.ny}), &buffer2_, nullptr);
            context->allocate_persistent(DT_DOUBLE, TensorShape({param_.na*2}), &buffer3_, nullptr);
        } else {
            context->CtxFailure(__FILE__, __LINE__, 
                errors::InvalidArgument("Unrecognised projection method. \
                    Only pixel driven is available for fan geometry now."));
        }
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        OP_REQUIRES(context, input.dims() == 4, errors::InvalidArgument("Input must be 4-dimensional."));
        const int64 in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
        const int64 in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
        const int64 in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
        const int64 in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
        OP_REQUIRES(context, in_height == static_cast<int64>(param_.ny), 
                    errors::InvalidArgument("Input height must equal to that given in proj_shape."));
        OP_REQUIRES(context, in_width == static_cast<int64>(param_.nx), 
                    errors::InvalidArgument("Input width must equal to that given in proj_shape."));
        OP_REQUIRES(context, in_channel == 1, 
                    errors::InvalidArgument("Input channel number must be 1."));
        
        // initialize if needed
        if (!initialized_) {
            LaunchBpPar2DGradPrepOp<Device>()(context, 
                buffer1_.AccessTensor(context)->template flat<double>().data(), 
                buffer2_.AccessTensor(context)->template flat<double>().data(), 
                buffer3_.AccessTensor(context)->template flat<double>().data(), 
                grad_prep_.get());
            initialized_ = true;
        }

        // calculate results
        TensorShape out_shape({in_batch, param_.na, param_.ns, 1});
        Tensor* output = nullptr;
        // allocate result tensor
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
        LaunchBpPar2DGradOp<Device, T>()(context, input.template flat<T>().data(), 
            output->template flat<T>().data(), 
            buffer1_.AccessTensor(context)->template flat<double>().data(), 
            buffer2_.AccessTensor(context)->template flat<double>().data(), 
            buffer3_.AccessTensor(context)->template flat<double>().data(), 
            gradient_.get(), in_batch, param_.nx*param_.ny, param_.ns*param_.na);
        
        return;
    }

private:
    // parameters and functors
    ct_recon::FanBackprojection2DParam param_;
    std::unique_ptr<ct_recon::FanBackprojection2DGradPrep> grad_prep_;
    std::unique_ptr<ct_recon::FanBackprojection2DGrad<T>> gradient_;
    bool initialized_;

    // tensor buffers
    PersistentTensor buffer1_;
    PersistentTensor buffer2_;
    PersistentTensor buffer3_;
}; // class BackprojectionFan2DGradOp

// register class to operator
REGISTER_CPU_KERNEL("BackprojectionFan2D", float, BackprojectionFan2DOp)
REGISTER_CPU_KERNEL("BackprojectionFan2D", double, BackprojectionFan2DOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("BackprojectionFan2D", float, BackprojectionFan2DOp)
REGISTER_GPU_KERNEL("BackprojectionFan2D", double, BackprojectionFan2DOp)
#endif

REGISTER_CPU_KERNEL("BackprojectionFan2DGrad", float, BackprojectionFan2DGradOp)
REGISTER_CPU_KERNEL("BackprojectionFan2DGrad", double, BackprojectionFan2DGradOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("BackprojectionFan2DGrad", float, BackprojectionFan2DGradOp)
REGISTER_GPU_KERNEL("BackprojectionFan2DGrad", double, BackprojectionFan2DGradOp)
#endif

} // namespace tensorflow