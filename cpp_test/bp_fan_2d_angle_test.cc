/*
 * @Description: 
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 13:15:48
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 13:32:49
 */


#include "include/bp_fan_2d_angle.h"
#include "include/filter.h"
#include "include/fan_weighting.h"

#define USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>

int main()
{
    // parameter setting
    unsigned int ns = 848;
    unsigned int na = 720;
    double ds = 0.6;
    double orbit = 0.5 * M_PI / 180;
    double offset_s = 0.0;
    double orbit_start = 0.0;
    unsigned int nx = 512;
    unsigned int ny = 512;
    double dx = 0.4;
    double dy = 0.4;
    double offset_x = 0.0;
    double offset_y = 0.0;
    double dso = 870;
    double dsd = 1270;
    double fov = -1;
    ct_recon::FanWeightingParam wparam(ns, na/2, ds, dso, dsd, "fan");
    ct_recon::FilterParam fparam(ns, na/2, ds, dsd, "fan");
    ct_recon::FanBackprojection2DAngleParam param(ns, na/2, ds, offset_s, 
        nx, ny, dx, dy, offset_x, offset_y, dso, dsd, fov);
    
    // load projection data
    std::fstream f_in("../../fan_proj.raw", std::ios::in|std::ios::binary);
    if (!f_in.is_open()) {
        std::cout << "Unable to open input file!" << std::endl;
        return 1;
    }
    int* pData = new int[ns*na];
    f_in.read(reinterpret_cast<char*>(pData), ns*na*sizeof(int));
    f_in.close();
    double *proj_full = new double[ns*na];
    for (int i = 0; i < ns*na; ++i) proj_full[i] = static_cast<double>(pData[i]);
    delete[] pData;
    double* proj_part = new double[ns*na/2];
    int ia_new = 0;
    double* angles = new double[na/2];
    for (int ia = 0; ia < na; ++ia) {
        if ((ia/90) % 2 == 1) continue;
        angles[ia_new] = orbit_start + static_cast<double>(ia) * orbit;
        for (int is = 0; is < ns; ++is)
            proj_part[is+ia_new*ns] = proj_full[is+ia*ns];
        ++ia_new;
    }
    delete[] proj_full;

    // allocate temporary arrays
    double* xpos = new double[nx];
    double* ypos = new double[ny];
    double* sincostbl = new double[na];
    double* img = new double[nx*ny];
    double* filter = new double[2*ns+1];
    double* pFiltered = new double[ns*na / 2];

    ct_recon::FanWeighting<double> wght(wparam);
    ct_recon::RampFilterPrep<double> filt_prep(fparam);
    ct_recon::RampFilter<double> filt(fparam);
    ct_recon::FanBackprojection2DAnglePixDrivenPrep fbp_prep(param);
    ct_recon::FanBackprojection2DAnglePixDriven<double> fbp(param);

    // compute
    bool succ;
    filt_prep.calculate_on_cpu(filter);
    succ = fbp_prep.calculate_on_cpu(angles, xpos, ypos, sincostbl);
    if (succ) {
        // fbp
        std::clock_t begin = clock();
        wght.calculate_on_cpu(proj_part, proj_part);
        filt.calculate_on_cpu(proj_part, filter, pFiltered);
        succ = fbp.calculate_on_cpu(pFiltered, img, xpos, ypos, sincostbl);
        std::clock_t duration = clock() - begin;
        std::cout<<"cost "<<static_cast<double>(duration) / CLOCKS_PER_SEC<<" seconds"<<std::endl;
        if (succ) {
            // save result
            std::fstream f_out("../../fan_recon_part1.raw", std::ios::out|std::ios::binary);
            if (f_out.is_open()) {
                f_out.write(reinterpret_cast<char*>(img), nx*ny*sizeof(double));
                f_out.close();
            } else {
                std::cout << "Unable to open output file!" << std::endl;
            }
        }
    }

    // free allocated memory
    delete[] proj_part;
    delete[] xpos;
    delete[] ypos;
    delete[] sincostbl;
    delete[] img;
    delete[] filter;
    delete[] pFiltered;
    return 0;
}