/*
 * @Author: Tianling Lyu
 * @Date: 2021-02-05 11:15:46
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-02-07 16:22:40
 * @FilePath: /CT_RECON_EXTENSIONS/tensorflow/fan_weighting_ops.cu
 */

#include "tensorflow/fan_weighting_ops.h"

#include <vector>
#include <memory>
#include <cstdio>
#include <exception>
 
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
 
#include "include/fan_weighting.h"

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
REGISTER_OP("FanWeighting")
    .Input("proj_in: T")
    .Output("proj_out: T")
    .Attr("T: { float, double }")
    .Attr("proj_shape: list(int) >= 2")
    .Attr("channel_space: float")
    .Attr("channel_offset: float")
    .Attr("dso: float")
    .Attr("dsd: float")
    .Attr("tp: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));
        context->set_output(0, input_shape);
        return Status::OK();
    });

REGISTER_OP("FanWeightingGrad")
    .Input("proj_in: T")
    .Output("proj_out: T")
    .Attr("T: { float, double }")
    .Attr("proj_shape: list(int) >= 2")
    .Attr("channel_space: float")
    .Attr("channel_offset: float")
    .Attr("dso: float")
    .Attr("dsd: float")
    .Attr("tp: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));
        context->set_output(0, input_shape);
        return Status::OK();
    });

template <typename Device, typename T>
class FanWeightingOp : public OpKernel {
    public:
        FanWeightingOp(OpKernelConstruction* context)
            : OpKernel(context)
        {
            // load and check parameters
            ::std::vector<int> proj_shape;
            OP_REQUIRES_OK(context, context->GetAttr("proj_shape", &proj_shape));
            OP_REQUIRES(context, proj_shape.size() == 2, 
                        errors::InvalidArgument("Proj_shape must contain 2 values."));
            OP_REQUIRES(context, proj_shape.at(0) > 0 && proj_shape.at(1) > 0, 
                        errors::InvalidArgument("Proj_shape must be positive integers. "));
            float channel_space, channel_offset;
            OP_REQUIRES_OK(context, context->GetAttr("channel_space", &channel_space));
            OP_REQUIRES_OK(context, context->GetAttr("channel_offset", &channel_offset));
            float dso, dsd;
            OP_REQUIRES_OK(context, context->GetAttr("dso", &dso));
            OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
            ::std::string type;
            OP_REQUIRES_OK(context, context->GetAttr("tp", &type));
            int itype = -1;
            if (type == "fan") itype = 1;
            else if (type == "flat") itype = 2;
            // construct private parameters and functors
            param_ = ct_recon::FanWeightingParam(proj_shape[1], proj_shape[0], 
                channel_space, channel_offset, dso, dsd, itype);
            if (type != "fan" && type != "flat") {
                context->CtxFailure(__FILE__, __LINE__,
                                    errors::InvalidArgument("Unrecognised geometry type. \
                        Only fan or flat are available."));
            }
            fw_ = std::unique_ptr<ct_recon::FanWeighting<T>>
                (new ct_recon::FanWeighting<T>(param_));
        }
    
        void Compute(OpKernelContext* context) override {
            const Tensor& input = context->input(0);
            OP_REQUIRES(context, input.dims() == 4, errors::InvalidArgument("Input must be 4-dimensional. "));
            const int64 in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            const int64 in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            const int64 in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            const int64 in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height == static_cast<int64>(param_.nrow), 
                        errors::InvalidArgument("Input height must equal to that given in proj_shape. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in proj_shape. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
    
            // calculate results
            TensorShape out_shape({in_batch, param_.nrow, param_.ns, 1});
            Tensor* output = nullptr;
            // allocate result tensor
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
            bool ok = LaunchFanWOp<Device, T>()(context, 
                input.template flat<T>().data(), 
                output->template flat<T>().data(), fw_.get(), in_batch, 
                param_.ns*param_.nrow);
            return;
        }
    
    private:
        // parameters and functors
        ct_recon::FanWeightingParam param_;
        std::unique_ptr<ct_recon::FanWeighting<T>> fw_;
}; // class FanWeightingOp

template <typename Device, typename T>
class FanWeightingGradOp : public OpKernel {
    public:
        FanWeightingGradOp(OpKernelConstruction* context)
            : OpKernel(context)
        {
            // load and check parameters
            ::std::vector<int> proj_shape;
            OP_REQUIRES_OK(context, context->GetAttr("proj_shape", &proj_shape));
            OP_REQUIRES(context, proj_shape.size() == 2, 
                        errors::InvalidArgument("Proj_shape must contain 2 values."));
            OP_REQUIRES(context, proj_shape.at(0) > 0 && proj_shape.at(1) > 0, 
                        errors::InvalidArgument("Proj_shape must be positive integers. "));
            float channel_space, channel_offset;
            OP_REQUIRES_OK(context, context->GetAttr("channel_space", &channel_space));
            OP_REQUIRES_OK(context, context->GetAttr("channel_offset", &channel_offset));
            float dso, dsd;
            OP_REQUIRES_OK(context, context->GetAttr("dso", &dso));
            OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
            ::std::string type;
            OP_REQUIRES_OK(context, context->GetAttr("tp", &type));
            int itype = -1;
            if (type == "fan") itype = 1;
            else if (type == "flat") itype = 2;
            // construct private parameters and functors
            param_ = ct_recon::FanWeightingParam(proj_shape[1], proj_shape[0], 
                channel_space, channel_offset, dso, dsd, itype);
            if (type != "fan" && type != "flat") {
                context->CtxFailure(__FILE__, __LINE__,
                                    errors::InvalidArgument("Unrecognised geometry type. \
                        Only fan or flat are available."));
            }
            fw_grad_ = std::unique_ptr<ct_recon::FanWeightingGrad<T>>
                (new ct_recon::FanWeightingGrad<T>(param_));
        }
    
        void Compute(OpKernelContext* context) override {
            const Tensor& input = context->input(0);
            OP_REQUIRES(context, input.dims() == 4, errors::InvalidArgument("Input must be 4-dimensional. "));
            const int64 in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            const int64 in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            const int64 in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            const int64 in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height == static_cast<int64>(param_.nrow), 
                        errors::InvalidArgument("Input height must equal to that given in proj_shape. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in proj_shape. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
    
            // calculate results
            TensorShape out_shape({in_batch, param_.nrow, param_.ns, 1});
            Tensor* output = nullptr;
            // allocate result tensor
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
            bool ok = LaunchFanWGradOp<Device, T>()(context, 
                input.template flat<T>().data(), 
                output->template flat<T>().data(), fw_grad_.get(), in_batch, 
                param_.ns*param_.nrow);
            return;
        }
    
    private:
        // parameters and functors
        ct_recon::FanWeightingParam param_;
        std::unique_ptr<ct_recon::FanWeightingGrad<T>> fw_grad_;
}; // class FanWeightingGradOp

// register class to operator
REGISTER_CPU_KERNEL("FanWeighting", float, FanWeightingOp)
REGISTER_CPU_KERNEL("FanWeighting", double, FanWeightingOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("FanWeighting", float, FanWeightingOp)
REGISTER_GPU_KERNEL("FanWeighting", double, FanWeightingOp)
#endif

REGISTER_CPU_KERNEL("FanWeightingGrad", float, FanWeightingGradOp)
REGISTER_CPU_KERNEL("FanWeightingGrad", double, FanWeightingGradOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("FanWeightingGrad", float, FanWeightingGradOp)
REGISTER_GPU_KERNEL("FanWeightingGrad", double, FanWeightingGradOp)
#endif

} // namespace tensorflow