/*
 * @Description: pytorch operators for 2-D parallel beam backprojection
 * @Author: Tianling Lyu
 * @Date: 2020-02-06 10:21:01
 * @LastEditors  : Tianling Lyu
 * @LastEditTime : 2020-02-10 09:31:35
 */

#include <vector>
#include <string>
#include <stdexcept>

#include <torch/extension.h>

#include "include/bp_par_2d.h"

std::vector<torch::Tensor> bppar2d_prep_forward(std::string method, 
                                                unsigned int ns,
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
{
    ct_recon::ParallelBackprojection2DParam param(ns, na, ds, orbit, offset_s, 
        orbit_start, nx, ny, dx, dy, offset_x, offset_y, fov);
    
    if (method == "pixdriven") {
        auto buffer1 = torch::zeros({nx, na*sizeof(double)});
        auto buffer2 = torch::zeros({ny, na*sizeof(double)});
        auto buffer3 = torch::zeros({sizeof(int)});

        prep = ct_recon::ParallelBackprojection2DPixDrivenPrep(param);
        prep.calculate_on_cpu(buffer1.data<double>(), buffer.data<double>(), buffer3.data<int>());

        return { buffer1, buffer2, buffer3 };
    } else {
        throw std::runtime_error("Unrecognised backprojection method. \
                    Only pixdriven is available now.");
    }
}

torch::Tensor bppar2d_forward(torch::Tensor input,
                              torch::Tensor buffer1,
                              torch::Tensor buffer2,
                              torch::Tensor buffer3,
                              std::string method,
                              unsigned int ns,
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
{
    ct_recon::ParallelBackprojection2DParam param(ns, na, ds, orbit, offset_s,
                                                  orbit_start, nx, ny, dx, dy, offset_x, offset_y, fov);
    
    auto proj = torch::zeros({});
}