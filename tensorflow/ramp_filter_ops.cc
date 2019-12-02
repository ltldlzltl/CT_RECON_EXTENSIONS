/*
 * @Description: ramp filter operator for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-12-02 09:15:47
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-02 14:40:01
 */

#include "tensorflow/ramp_filter_ops.h"

#include <vector>
#include <memory>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"

#include "include/filter.h"

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
REGISTER_OP("RampFilter")
    .Input("inproj: T")
    .Output("out: T")
    .Attr("T: { float, double }")
    .Attr("ns: int")
    .Attr("nrow: int")
    .Attr("ds: float")
    .Attr("dsd: float")
    .Attr("type: string")
    .Attr("window: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        return shape_inference::UnchangedShapeWithRankAtLeast(context, 4);
    });

REGISTER_OP("RampFilterGrad")
    .Input("inimg: T")
    .Output("out: T")
    .Attr("T: { float, double }")
    .Attr("ns: int")
    .Attr("nrow: int")
    .Attr("ds: float")
    .Attr("dsd: float")
    .Attr("type: string")
    .Attr("window: string")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
        return shape_inference::UnchangedShapeWithRankAtLeast(context, 4);
    });

template <typename Device, typename T>
class RampFilterOp : public OpKernel {
public:
    RampFilterOp(OpKernelConstruction* context)
        : OpKernel(context), initialized_(false)
    {
        // load parameters
        int ns, nrow;
        float ds, dsd;
        std::string type, window;
        OP_REQUIRES_OK(context, context->GetAttr("ns", &ns));
        OP_REQUIRES_OK(context, context->GetAttr("nrow", &nrow));
        OP_REQUIRES_OK(context, context->GetAttr("ds", &ds));
        OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
        OP_REQUIRES_OK(context, context->GetAttr("type", &type));
        OP_REQUIRES_OK(context, context->GetAttr("window", &window));
        param_ = ct_recon::FilterParam(ns, nrow, ds, dsd, type, window);
        // construct functors
        prep_ = std::unique_ptr<ct_recon::RampFilterPrep<T>>
            (new ct_recon::RampFilterPrep<T>(param_));
        filt_ = std::unique_ptr<ct_recon::RampFilter<T>>
            (new ct_recon::RampFilter<T>(param_));
    }

    void Compute(OpKernelContext* context) override {
        // check input shape
        const Tensor& input = context->input(0);
        int ndims = input.dims();
        OP_REQUIRES(context, ndims >= 4, errors::InvalidArgument("Input must be at least 4-D."));
        int64 in_batch, in_depth, in_height, in_width, in_channel;
        unsigned int sizeproj;
        if (ndims == 4) {
            in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height == static_cast<int64>(param_.nrow), 
                        errors::InvalidArgument("Input height must equal to that given in nrow. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in ns. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
            sizeproj = in_height * in_width;
        } else {
            in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            in_depth = GetTensorDim(input, FORMAT_NHWC, '0');
            in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height*in_depth == static_cast<int64>(param_.nrow), 
                        errors::InvalidArgument("Input height must equal to that given. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in ns. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
            sizeproj = in_depth * in_height * in_width;
        }

        // initialize if needed
        if (!initialized_) {
            LaunchRampFilterPrepOp<Device, T>()(context, 
                filter_.AccessTensor(context)->template flat<T>().data(), 
                prep_.get());
            initialized_ = true;
        }

        // allocate output tensor
        TensorShape out_shape(input.shape());
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
        // calculate results
        LaunchRampFilterOp<Device, T>()(context, input.template flat<T>().data(), 
            filter_.AccessTensor(context)->template flat<T>().data(), 
            output->template flat<T>().data(), filt_.get(), in_batch, 
            sizeproj);
        return;
    }

private:
    // parameters and functors
    ct_recon::FilterParam param_;
    std::unique_ptr<ct_recon::RampFilterPrep<T>> prep_;
    std::unique_ptr<ct_recon::RampFilter<T>> filt_;
    bool initialized_;
    // buffer
    PersistentTensor filter_;
}; // class RampFilterOp

template <typename Device, typename T>
class RampFilterGradOp : public OpKernel {
public:
    RampFilterGradOp(OpKernelConstruction* context)
        : OpKernel(context), initialized_(false)
    {
        // load parameters
        int ns, nrow;
        float ds, dsd;
        std::string type, window;
        OP_REQUIRES_OK(context, context->GetAttr("ns", &ns));
        OP_REQUIRES_OK(context, context->GetAttr("nrow", &nrow));
        OP_REQUIRES_OK(context, context->GetAttr("ds", &ds));
        OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
        OP_REQUIRES_OK(context, context->GetAttr("type", &type));
        OP_REQUIRES_OK(context, context->GetAttr("window", &window));
        param_ = ct_recon::FilterParam(ns, nrow, ds, dsd, type, window);
        // construct functors
        prep_ = std::unique_ptr<ct_recon::RampFilterPrep<T>>
            (new ct_recon::RampFilterPrep<T>(param_));
        filt_ = std::unique_ptr<ct_recon::RampFilterGrad<T>>
            (new ct_recon::RampFilterGrad<T>(param_));
    }

    void Compute(OpKernelContext* context) override {
        // check input shape
        const Tensor& input = context->input(0);
        int ndims = input.dims();
        OP_REQUIRES(context, ndims >= 4, errors::InvalidArgument("Input must be at least 4-D."));
        int64 in_batch, in_depth, in_height, in_width, in_channel;
        unsigned int sizeproj;
        if (ndims == 4) {
            in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height == static_cast<int64>(param_.nrow), 
                        errors::InvalidArgument("Input height must equal to that given in nrow. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in ns. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
            sizeproj = in_height * in_width;
        } else {
            in_batch = GetTensorDim(input, FORMAT_NHWC, 'N');
            in_depth = GetTensorDim(input, FORMAT_NHWC, '0');
            in_height = GetTensorDim(input, FORMAT_NHWC, 'H');
            in_width = GetTensorDim(input, FORMAT_NHWC, 'W');
            in_channel = GetTensorDim(input, FORMAT_NHWC, 'C');
            OP_REQUIRES(context, in_height*in_depth == static_cast<int64>(param_.nrow), 
                        errors::InvalidArgument("Input height must equal to that given. "));
            OP_REQUIRES(context, in_width == static_cast<int64>(param_.ns), 
                        errors::InvalidArgument("Input width must equal to that given in ns. "));
            OP_REQUIRES(context, in_channel == 1, 
                        errors::InvalidArgument("Input channel number must be 1. "));
            sizeproj = in_depth * in_height * in_width;
        }

        // initialize if needed
        if (!initialized_) {
            LaunchRampFilterPrepOp<Device, T>()(context, 
                filter_.AccessTensor(context)->template flat<T>().data(), 
                prep_.get());
            initialized_ = true;
        }

        // allocate output tensor
        TensorShape out_shape(input.shape());
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
        // calculate results
        LaunchRampFilterGradOp<Device, T>()(context, input.template flat<T>().data(), 
            filter_.AccessTensor(context)->template flat<T>().data(), 
            output->template flat<T>().data(), filt_.get(), in_batch, 
            sizeproj);
        return;
    }

private:
    // parameters and functors
    ct_recon::FilterParam param_;
    std::unique_ptr<ct_recon::RampFilterPrep<T>> prep_;
    std::unique_ptr<ct_recon::RampFilterGrad<T>> filt_;
    bool initialized_;
    // buffer
    PersistentTensor filter_;
}; // class RampFilterGradOp

// register class to operator
REGISTER_CPU_KERNEL("RampFilter", float, RampFilterOp)
REGISTER_CPU_KERNEL("RampFilter", double, RampFilterOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("RampFilter", float, RampFilterOp)
REGISTER_GPU_KERNEL("RampFilter", double, RampFilterOp)
#endif

REGISTER_CPU_KERNEL("RampFilterGrad", float, RampFilterGradOp)
REGISTER_CPU_KERNEL("RampFilterGrad", double, RampFilterGradOp)
#if GOOGLE_CUDA
REGISTER_GPU_KERNEL("RampFilterGrad", float, RampFilterGradOp)
REGISTER_GPU_KERNEL("RampFilterGrad", double, RampFilterGradOp)
#endif

} // namespace tensorflow