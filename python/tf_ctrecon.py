"""
" @Description: Python tensorflow extension of CT reconstruction
" @Author: Tianling Lyu
" @Date: 2019-12-03 13:47:00
 # @LastEditors: Tianling Lyu
 # @LastEditTime: 2021-03-11 09:43:47
"""

import math
import tensorflow as tf
from tensorflow.python.framework import ops

# load dynamic library
_ct_ops_module = tf.load_op_library("build/lib/libtensorflow_ctext.so")
#_ct_ops_module = tf.load_op_library("lib/libctrecon.so")
_rampfilter = _ct_ops_module.ramp_filter
_ramp_grad = _ct_ops_module.ramp_filter_grad
_bp_par = _ct_ops_module.backprojection_parallel2d
_bp_par_grad = _ct_ops_module.backprojection_parallel2d_grad
_fp_par = _ct_ops_module.forward_projection_parallel2d
_fp_par_grad = _ct_ops_module.forward_projection_parallel2d_grad
_svbp_par = _ct_ops_module.single_view_bp_parallel2d
_bp_fan = _ct_ops_module.backprojection_fan2d
_bp_fan_grad = _ct_ops_module.backprojection_fan2d_grad
_fanw = _ct_ops_module.fan_weighting
_fanw_grad = _ct_ops_module.fan_weighting_grad

# register gradients
@ops.RegisterGradient("RampFilter")
def _ramp_filter_grad(op, grad):
    ns = op.get_attr("ns")
    nrow = op.get_attr("nrow")
    ds = op.get_attr("ds")
    dsd = op.get_attr("dsd")
    ftype = op.get_attr("type")
    window = op.get_attr("window")
    in_grad = _ramp_grad(inimg=grad, ns=ns, nrow=nrow, ds=ds, dsd=dsd, type=ftype, 
        window=window)
    return [in_grad]

@ops.RegisterGradient("BackprojectionParallel2D")
def _backprojection_parallel2d_grad(op, grad):
    img_shape = op.get_attr("img_shape")
    img_space = op.get_attr("img_space")
    img_offset = op.get_attr("img_offset")
    proj_shape = op.get_attr("proj_shape")
    channel_space = op.get_attr("channel_space")
    channel_offset = op.get_attr("channel_offset")
    orbit_start = op.get_attr("orbit_start")
    orbit = op.get_attr("orbit")
    fov = op.get_attr("fov")
    method = op.get_attr("method")
    in_grad = _bp_par_grad(img=grad, img_shape=img_shape, img_space=img_space, 
            img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
            channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
            fov=fov, method=method)
    return [in_grad]

@ops.RegisterGradient("ForwardProjectionParallel2D")
def _forward_projection_parallel2d_grad(op, grad):
    img_shape = op.get_attr("img_shape")
    img_space = op.get_attr("img_space")
    img_offset = op.get_attr("img_offset")
    proj_shape = op.get_attr("proj_shape")
    channel_space = op.get_attr("channel_space")
    channel_offset = op.get_attr("channel_offset")
    orbit_start = op.get_attr("orbit_start")
    orbit = op.get_attr("orbit")
    fov = op.get_attr("fov")
    method = op.get_attr("method")
    in_grad = _fp_par_grad(proj=grad, img_shape=img_shape, img_space=img_space, 
            img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
            channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
            fov=fov, method=method)
    return [in_grad]

@ops.RegisterGradient("BackprojectionFan2D")
def _fan_backprojection2d_grad(op, grad):
    img_shape = op.get_attr("img_shape")
    img_space = op.get_attr("img_space")
    img_offset = op.get_attr("img_offset")
    proj_shape = op.get_attr("proj_shape")
    channel_space = op.get_attr("channel_space")
    channel_offset = op.get_attr("channel_offset")
    orbit_start = op.get_attr("orbit_start")
    orbit = op.get_attr("orbit")
    dso = op.get_attr("dso")
    dsd = op.get_attr("dsd")
    fov = op.get_attr("fov")
    method = op.get_attr("method")
    in_grad = _bp_fan_grad(img=grad, img_shape=img_shape, img_space=img_space, 
            img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
            channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
            dso=dso, dsd=dsd, fov=fov, method=method)
    return [in_grad]

@ops.RegisterGradient("FanWeighting")
def _fan_weighting_grad(op, grad):
    proj_shape = op.get_attr("proj_shape")
    channel_space = op.get_attr("channel_space")
    channel_offset = op.get_attr("channel_offset")
    dso = op.get_attr("dso")
    dsd = op.get_attr("dsd")
    tp = op.get_attr("tp")
    in_grad = _fanw_grad(proj_in=grad, proj_shape=proj_shape, 
        channel_space=channel_space, channel_offset=channel_offset, dso=dso,
        dsd=dsd, tp=tp)
    return [in_grad]

# function wrappers

def ramp_filter(proj, n_channel, n_row, channel_space, dsd=-1, type="par", 
                window="None"):
    """
    Ramp filter function. Returns a sinogram the same size as proj.
        :param proj: input sinogram, 4-D or 5-D, NHWC format, channel number 
                     must be 1
        :param n_channel: number of channels on the detector
        :param n_row: total number of rows in the sinogram. 
        :param channel_space: distance between nearby channels. 
        :param dsd=-1: distance between projection source and detector, only
                       useful for "fan" type.
        :param type="par": "par" for parallel beam, "fan" for equiangular 
                            fanbeam, "flat" for equispacing fanbeam. 
        :param window="None": apply window on filter, not implemented yet. 
    """
    return _rampfilter(inproj=proj, ns=n_channel, nrow=n_row, ds=channel_space, 
        dsd=dsd, type=type, window=window)

def ramp_filter_grad(grad_in, n_channel, n_row, channel_space, dsd=-1, type="par", 
                window="None"):
    """
    Ramp filter function. Returns a sinogram the same size as proj.
        :param grad_in: input gradient, 4-D or 5-D, NHWC format, channel number 
                     must be 1
        :param n_channel: number of channels on the detector
        :param n_row: total number of rows in the sinogram. 
        :param channel_space: distance between nearby channels. 
        :param dsd=-1: distance between projection source and detector, only
                       useful for "fan" type.
        :param type="par": "par" for parallel beam, "fan" for equiangular 
                            fanbeam, "flat" for equispacing fanbeam. 
        :param window="None": apply window on filter, not implemented yet. 
    """
    return _ramp_grad(inimg=grad_in, ns=n_channel, nrow=n_row, ds=channel_space, 
        dsd=dsd, type=type, window=window)


def parallel_backprojection_2d(proj, img_shape, img_space, proj_shape, 
    channel_space, img_offset=[0, 0], channel_offset=0, orbit_start=0, 
    orbit=math.pi/180, fov=-1, method='pixdriven'):
    """
    2-D parallel backprojection function. 
        :param proj: input sinogram, [batch, nview, nchannel, 1]
        :param img_shape: list(int) with length=2. Height and width of result.
        :param img_space: list(int) with length=2. Spacing between image pixels.
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param img_offset=[0, 0]: difference between ISO center and 
                                   [(nx-1)/2, (ny-1)/2].
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param orbit_start=0: angle at the first view
        :param orbit=math.pi/180: rotated angle between nearby views
        :param fov=-1: field of view, useless now. 
        :param method='pixdriven': bp operator, only 'pixdriven' is available now.
    """
    return _bp_par(proj=proj, img_shape=img_shape, img_space=img_space, 
        img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
        fov=fov, method=method)


def parallel_projection_2d(img, img_shape, img_space, proj_shape, 
    channel_space, img_offset=[0, 0], channel_offset=0, orbit_start=0, 
    orbit=math.pi/180, fov=-1, method='pixdriven'):
    """
    docstring here
        :param img: input image, [nbatch, height, width, 1]
        :param img_shape: list(int) with length=2. Height and width of result.
        :param img_space: list(int) with length=2. Spacing between image pixels.
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param img_offset=[0, 0]: difference between ISO center and 
                                   [(nx-1)/2, (ny-1)/2].
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param orbit_start=0: angle at the first view
        :param orbit=math.pi/180: rotated angle between nearby views
        :param fov=-1: field of view, only useful for "raycasting". 
        :param method='pixdriven': fp operator, can be 'raycasting' or 
            'raydriven'.
    """
    return _fp_par(img=img, img_shape=img_shape, img_space=img_space, 
        img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
        fov=fov, method=method)


def parallel_single_view_bp_2d(proj, img_shape, img_space, proj_shape, 
    channel_space, img_offset=[0, 0], channel_offset=0, orbit_start=0, 
    orbit=math.pi/180, fov=-1, method='pixdriven'):
    """
    docstring here
        :param proj: input sinogram, [batch, nview, nchannel, 1]
        :param img_shape: list(int) with length=2. Height and width of result.
        :param img_space: list(int) with length=2. Spacing between image pixels.
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param img_offset=[0, 0]: difference between ISO center and 
                                   [(nx-1)/2, (ny-1)/2].
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param orbit_start=0: angle at the first view
        :param orbit=math.pi/180: rotated angle between nearby views
        :param fov=-1: field of view, useless now. 
        :param method='pixdriven': bp operator, only 'pixdriven' is available now.
    """
    return _svbp_par(proj=proj, img_shape=img_shape, img_space=img_space, 
        img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
        fov=fov, method=method)

def fan_backprojection_2d(proj, img_shape, img_space, proj_shape, 
    channel_space, img_offset=[0, 0], channel_offset=0, orbit_start=0, 
    orbit=math.pi/180, dso=1000, dsd=1500, fov=-1, method='pixdriven'):
    """
    2-D parallel backprojection function. 
        :param proj: input sinogram, [batch, nview, nchannel, 1]
        :param img_shape: list(int) with length=2. Height and width of result.
        :param img_space: list(int) with length=2. Spacing between image pixels.
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param img_offset=[0, 0]: difference between ISO center and 
                                   [(nx-1)/2, (ny-1)/2].
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param orbit_start=0: angle at the first view
        :param orbit=math.pi/180: rotated angle between nearby views
        :param dso=1000: distance between source and ISO center (mm).
        :param dsd=1500: distance between source and detector center (mm).
        :param fov=-1: field of view, useless now. 
        :param method='pixdriven': bp operator, only 'pixdriven' is available now.
    """
    return _bp_fan(proj=proj, img_shape=img_shape, img_space=img_space, 
        img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
        dso=dso, dsd=dsd, fov=fov, method=method)

def fan_backprojection_2d_grad(grad_in, img_shape, img_space, proj_shape, 
    channel_space, img_offset=[0, 0], channel_offset=0, orbit_start=0, 
    orbit=math.pi/180, dso=1000, dsd=1500, fov=-1, method='pixdriven'):
    """
    2-D parallel backprojection function. 
        :param grad_in: input gradient, [batch, nview, nchannel, 1]
        :param img_shape: list(int) with length=2. Height and width of result.
        :param img_space: list(int) with length=2. Spacing between image pixels.
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param img_offset=[0, 0]: difference between ISO center and 
                                   [(nx-1)/2, (ny-1)/2].
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param orbit_start=0: angle at the first view
        :param orbit=math.pi/180: rotated angle between nearby views
        :param dso=1000: distance between source and ISO center (mm).
        :param dsd=1500: distance between source and detector center (mm).
        :param fov=-1: field of view, useless now. 
        :param method='pixdriven': bp operator, only 'pixdriven' is available now.
    """
    return _bp_fan_grad(img=grad_in, img_shape=img_shape, img_space=img_space, 
        img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
        dso=dso, dsd=dsd, fov=fov, method=method)

def fan_weighting(proj, proj_shape, channel_space, channel_offset=0, 
    dso=1000, dsd=1500, tp='fan'):
    """
    Fan weighting function. 
        :param proj: input sinogram, [batch, nview, nchannel, 1]
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param dso=1000: distance between source and ISO center (mm).
        :param dsd=1500: distance between source and detector center (mm).
        :param tp='fan': geometry type, "fan" or "flat".
    """
    return _fanw(proj_in=proj, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, dso=dso, dsd=dsd, tp=tp)

def fan_weighting_grad(grad_in, proj_shape, channel_space, channel_offset=0, 
    dso=1000, dsd=1500, tp='fan'):
    """
    Fan weighting function. 
        :param grad_in: input gradient, [batch, nview, nchannel, 1]
        :param proj_shape: list(int) with length=2. Input view number and channel number.
        :param channel_space: distance between nearby channels.
        :param channel_offset=0: difference between detector center and (ns-1)/2
        :param dso=1000: distance between source and ISO center (mm).
        :param dsd=1500: distance between source and detector center (mm).
        :param tp='fan': geometry type, "fan" or "flat".
    """
    return _fanw_grad(proj_in=grad_in, proj_shape=proj_shape, channel_space=channel_space, 
        channel_offset=channel_offset, dso=dso, dsd=dsd, tp=tp)