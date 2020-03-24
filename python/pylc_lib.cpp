#ifndef PYLC_LIB_HPP
#define PYLC_LIB_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pylc_converters.hpp>
#include <common.h>

using namespace lc;

PYBIND11_MODULE(pylc_lib, m) {

    // Datum Object
    py::class_<Datum, std::shared_ptr<Datum>>(m, "Datum")
    .def(py::init<>())
    .def_readwrite("type", &Datum::type)
    .def_readwrite("camera_name", &Datum::camera_name)
    .def_readwrite("rgb_matrix", &Datum::rgb_matrix)
    .def_readwrite("depth_matrix", &Datum::depth_matrix)
    .def_readwrite("world_to_rgb", &Datum::world_to_rgb)
    .def_readwrite("world_to_depth", &Datum::world_to_depth)
    .def_readwrite("cam_to_laser", &Datum::cam_to_laser)
    .def_readwrite("cam_to_world", &Datum::cam_to_world)
    .def_readwrite("fov", &Datum::fov)
    .def_readwrite("laser_name", &Datum::laser_name)
    .def_readwrite("distortion", &Datum::distortion)
    .def_readwrite("imgh", &Datum::imgh)
    .def_readwrite("imgw", &Datum::imgw)
    .def_readwrite("limit", &Datum::limit)
    .def_readwrite("galvo_m", &Datum::galvo_m)
    .def_readwrite("galvo_b", &Datum::galvo_b)
    .def_readwrite("maxADC", &Datum::maxADC)
    .def_readwrite("thickness", &Datum::thickness)
    .def_readwrite("divergence", &Datum::divergence)
    .def_readwrite("laser_limit", &Datum::laser_limit)
    .def_readwrite("laser_timestep", &Datum::laser_timestep)
    ;

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}

#endif