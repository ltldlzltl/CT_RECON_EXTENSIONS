/*
 * @Description: test fan 2-D backprojection functors
 * @Author: Tianling Lyu
 * @Date: 2021-01-07 10:56:33
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-02-07 16:58:50
 */

#include "include/bp_fan_2d.h"
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
    ct_recon::FanWeightingParam wparam(ns, na, ds, dso, dsd, 1);
    ct_recon::FilterParam fparam(ns, na, ds, dsd, 1);
    ct_recon::FanBackprojection2DParam param(ns, na, ds, orbit, offset_s, 
        orbit_start, nx, ny, dx, dy, offset_x, offset_y, dso, dsd, fov);
    
    // load projection data
    std::fstream f_in("../../fan_proj.raw", std::ios::in|std::ios::binary);
    if (!f_in.is_open()) {
        std::cout << "Unable to open input file!" << std::endl;
        return 1;
    }
    int* pData = new int[ns*na];
    f_in.read(reinterpret_cast<char*>(pData), ns*na*sizeof(int));
    f_in.close();
    double *pProj = new double[ns*na];
    for (int i = 0; i < ns*na; ++i) pProj[i] = static_cast<double>(pData[i]);
    delete[] pData;

    // allocate temporary arrays
    double* xpos = new double[nx];
    double* ypos = new double[ny];
    double* sincostbl = new double[na*2];
    double* img = new double[nx*ny];
    double* filter = new double[2*ns+1];
    double* pFiltered = new double[ns*na];

    ct_recon::FanWeighting<double> wght(wparam);
    ct_recon::RampFilterPrep<double> filt_prep(fparam);
    ct_recon::RampFilter<double> filt(fparam);
    ct_recon::FanBackprojection2DPixDrivenPrep fbp_prep(param);
    ct_recon::FanBackprojection2DPixDriven<double> fbp(param);

    // compute
    bool succ;
    filt_prep.calculate_on_cpu(filter);
    succ = fbp_prep.calculate_on_cpu(xpos, ypos, sincostbl);
    if (succ) {
        // fbp
        std::clock_t begin = clock();
        wght.calculate_on_cpu(pProj, pProj);
        filt.calculate_on_cpu(pProj, filter, pFiltered);
        succ = fbp.calculate_on_cpu(pFiltered, img, xpos, ypos, sincostbl);
        std::clock_t duration = clock() - begin;
        std::cout<<"cost "<<static_cast<double>(duration) / CLOCKS_PER_SEC<<" seconds"<<std::endl;
        if (succ) {
            // save result
            std::fstream f_filtered("../../fan_filtered.raw", std::ios::out|std::ios::binary);
            if (f_filtered.is_open()) {
                f_filtered.write(reinterpret_cast<char*>(pFiltered), ns*na*sizeof(double));
                f_filtered.close();
            } else {
                std::cout << "Unable to open filtered file!" << std::endl;
            }
            // save result
            std::fstream f_out("../../fan_recon.raw", std::ios::out|std::ios::binary);
            if (f_out.is_open()) {
                f_out.write(reinterpret_cast<char*>(img), nx*ny*sizeof(double));
                f_out.close();
            } else {
                std::cout << "Unable to open output file!" << std::endl;
            }
        }
    }

    // free allocated memory
    delete[] pProj;
    delete[] xpos;
    delete[] ypos;
    delete[] sincostbl;
    delete[] img;
    delete[] filter;
    delete[] pFiltered;
    return 0;
}