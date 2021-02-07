"""
 " @Description: load ctrecon cpp functions using numpy
 " @Author: Tianling Lyu
 " @Date: 2021-01-10 09:49:40
 # @LastEditors: Tianling Lyu
 # @LastEditTime: 2021-02-05 11:28:12
"""

import numpy as np
import numpy.ctypeslib as npct

# type defines
_array1dd = npct.ndpointer(np.double, 1, flags="C")
_array2dd = npct.ndpointer(np.double, 2, flags="C")
_uint = npct.ctypes.c_uint
_int = npct.ctypes.c_int
_double = npct.ctypes.c_double
_bool = npct.ctypes.c_bool

# load c++ library
_lib = npct.load_library("libnumpy_ctext.so", "build/lib")
# set device
_set_device = _lib.set_device
_set_device.rettype = _bool
_set_device.argtypes = [_int]
# clear device
_clear_device = _lib.clear
_clear_device.rettype = _bool
_clear_device.argtypes = []
# fan weighting
_fan_weighting_create = _lib.fan_weighting_create
_fan_weighting_create.rettype = _int
_fan_weighting_create.argtypes = [_uint, _uint, _double, _double, _double, _double, 
    _int]
_fan_weighting_run = _lib.fan_weighting_run
_fan_weighting_run.rettype = _bool
_fan_weighting_run.argtypes = [_int, _array2dd, _array2dd]
_fan_weighting_destroy = _lib.fan_weighting_destroy
_fan_weighting_destroy.rettype = _bool
_fan_weighting_destroy.argtypes = [_int]
# ramp filter
_ramp_filter_create = _lib.ramp_filter_create
_ramp_filter_create.rettype = _int
_ramp_filter_create.argtypes = [_uint, _uint, _double, _double, _int]
_ramp_filter_run = _lib.ramp_filter_run
_ramp_filter_run.rettype = _bool
_ramp_filter_run.argtypes = [_int, _array2dd, _array2dd]
_ramp_filter_destroy = _lib.ramp_filter_destroy
_ramp_filter_destroy.rettype = _bool
_ramp_filter_destroy.argtypes = [_int]
# fan bp 2d angle
_fan_bp_2d_angle_create = _lib.fan_bp_2d_angle_create
_fan_bp_2d_angle_create.rettype = _int
_fan_bp_2d_angle_create.argtypes = [_array1dd, _uint, _uint, _double, _double, 
    _uint, _uint, _double, _double, _double, _double, _double, _double, 
    _double]
_fan_bp_2d_angle_run = _lib.fan_bp_2d_angle_run
_fan_bp_2d_angle_run.rettype = _bool
_fan_bp_2d_angle_run.argtypes = [_int, _array2dd, _array2dd]
_fan_bp_2d_angle_destroy = _lib.fan_bp_2d_angle_destroy
_fan_bp_2d_angle_destroy.rettype = _bool
_fan_bp_2d_angle_destroy.argtypes = [_int]

# CT reconstruction classes

class FanFBP2DAngle:
    def __init__(self, param:dict):
        self._param = param
        itype = 1
        _set_device(param["device"])
        self._fw_handle = _fan_weighting_create(param["ns"], param["na"], 
            param["ds"], param["offset_s"], param["dso"], param["dsd"], itype)
        self._flt_handle = _ramp_filter_create(param["ns"], param["na"], 
            param["ds"], param["dsd"], itype)
        self._bp_handle = _fan_bp_2d_angle_create(param["angles"], param["ns"], 
            param["na"], param["ds"], param["offset_s"], param["nx"], 
            param["ny"], param["dx"], param["dy"], param["offset_x"], 
            param["offset_y"], param["dso"], param["dsd"], param["fov"])
    
    def __del__(self):
        _fan_weighting_destroy(self._fw_handle)
        _ramp_filter_destroy(self._flt_handle)
        _fan_bp_2d_angle_destroy(self._bp_handle)
    
    def run(self, proj):
        if proj.dtype != np.double:
            proj = proj.astype(np.double)
        recon = np.zeros([self._param["ny"], self._param["nx"]], dtype=np.double)
        na, ns = proj.shape
        if na != self._param["na"] or ns != self._param["ns"]:
            raise RuntimeError("Input shape not available!")
        filtered = np.zeros([self._param["na"], self._param["ns"]], dtype=np.double)
        if not _fan_weighting_run(self._fw_handle, proj, proj):
            raise RuntimeError("Fan weighting failed!")
        if not _ramp_filter_run(self._flt_handle, proj, filtered):
            raise RuntimeError("Ramp filter failed!")
        if not _fan_bp_2d_angle_run(self._bp_handle, filtered, recon):
            raise RuntimeError("Backprojection failed!")
        return recon * self._param["orbit"]