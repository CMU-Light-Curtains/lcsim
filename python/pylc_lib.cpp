#ifndef PYLC_LIB_HPP
#define PYLC_LIB_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pylc_converters.hpp>
#include <dprocessor.h>
#include <algo.h>
#include <fitting.h>
#include <planning.h>
#include <depth.h>

using namespace lc;

struct Sorted{
    std::string camera_name, laser_name;
    int input_index;
};

void processPointsJoint(std::shared_ptr<DatumProcessor>& datumProcessor, std::vector<std::shared_ptr<Input>>& inputs, py::dict& input_names, std::vector<py::dict>& soi, std::shared_ptr<Output>& output, bool get_cloud = false, bool get_full_cloud = false){

    // Resize
    output->clouds.resize(soi.size());
    output->images_multi.resize(soi.size());
    output->thickness_multi.resize(soi.size());

    // Resort
    std::vector<Sorted> sortedVec;
    sortedVec.resize(soi.size());
    for(int i=0; i<soi.size(); i++){
        std::string camera_name = std::string(py::str(soi[i]["C"]));
        std::string laser_name = std::string(py::str(soi[i]["L"]));
        int input_index = py::int_(input_names[camera_name.c_str()]);
        sortedVec[input_index].camera_name = camera_name;
        sortedVec[input_index].laser_name = laser_name;
        sortedVec[input_index].input_index = input_index;
    }
    pcl::PointCloud<pcl::PointXYZRGB> full_cloud;

    // Compute
    //#pragma omp parallel for shared(inputs, output, sortedVec)
    for(int i=0; i<sortedVec.size(); i++){
        Sorted& sorted = sortedVec[i];
        std::shared_ptr<Input>& input = inputs[i];

        // Get Depth Map
        cv::Mat depth_img = input->depth_image;

        // Resize local
        output->images_multi[i].resize(input->design_pts_multi.size());
        output->thickness_multi[i].resize(input->design_pts_multi.size());

        // Process
        pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;
        for(int j=0; j<input->design_pts_multi.size(); j++){
            const Eigen::MatrixXf& m = input->design_pts_multi[j];
            cv::Mat& image = output->images_multi[i][j];
            cv::Mat& thickness = output->thickness_multi[i][j];
            pcl::PointCloud<pcl::PointXYZRGB> cloud;
            datumProcessor->processPointsT(m, depth_img, sorted.camera_name, sorted.laser_name, image, thickness, cloud, get_cloud);
            combined_cloud += cloud;
        }

        // Convert pcl::PointCloud to Eigen::MatrixXf.
        output->clouds[i] = Eigen::Matrix<float, Eigen::Dynamic, 4> (combined_cloud.size(), 4);
        for (int pt_index = 0; pt_index < combined_cloud.size(); pt_index++) {
            const pcl::PointXYZRGB& point = combined_cloud[pt_index];
            output->clouds[i](pt_index, 0) = point.x;
            output->clouds[i](pt_index, 1) = point.y;
            output->clouds[i](pt_index, 2) = point.z;
            output->clouds[i](pt_index, 3) = point.g;
        }

        // Full
        if(get_full_cloud){
            pcl::PointCloud<pcl::PointXYZRGB> temp_cloud;
            Eigen::MatrixXf cam_to_world = datumProcessor->getCDatum(input->camera_name)->cam_to_world;
            pcl::transformPointCloud(combined_cloud, temp_cloud, cam_to_world);
            full_cloud += temp_cloud;
        }

    }

    if(get_full_cloud){

        // Remove temp
        pcl::PointCloud<pcl::PointXYZRGB> full_cloud_temp;
        for(auto i=0; i<full_cloud.size(); i++){
            const pcl::PointXYZRGB& p = full_cloud[i];
            if(std::isnan(p.x)) continue;
            full_cloud_temp.push_back(p);
        }
        full_cloud = full_cloud_temp;

        Eigen::MatrixXf full_cloud_eig;
        full_cloud_eig.resize(full_cloud.size(), 4);
        for(auto i=0; i<full_cloud.size(); i++){
            const pcl::PointXYZRGB& p = full_cloud[i];
            full_cloud_eig(i,0) = p.x;
            full_cloud_eig(i,1) = p.y;
            full_cloud_eig(i,2) = p.z;
            full_cloud_eig(i,3) = p.g;
        }
        output->full_cloud_eig = full_cloud_eig;

        #ifdef ROS
        pcl::toROSMsg(full_cloud, output->full_cloud);
        #endif
    }
}


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
            .def_readwrite("hit_N", &Datum::hit_N)
            .def_readwrite("hit_std", &Datum::hit_std)
            .def_readwrite("hit_pow", &Datum::hit_pow)
            .def_readwrite("hit_mode", &Datum::hit_mode)
            .def_readwrite("hit_noise", &Datum::hit_noise)
            .def_readwrite("valid_angles", &Datum::valid_angles)
            ;

    // Input Object
    py::class_<Input, std::shared_ptr<Input>>(m, "Input")
            .def(py::init<>())
            .def_readwrite("camera_name", &Input::camera_name)
            .def_readwrite("rgb_image", &Input::rgb_image)
            .def_readwrite("depth_image", &Input::depth_image)
            .def_readwrite("design_pts", &Input::design_pts)
            .def_readwrite("design_pts_multi", &Input::design_pts_multi)
            .def_readwrite("surface_pts", &Input::surface_pts)
            .def_readwrite("design_pts_conv", &Input::design_pts_conv)
            ;

    // Output Object
    py::class_<Output, std::shared_ptr<Output>>(m, "Output")
            .def(py::init<>())
            .def_readwrite("clouds", &Output::clouds)
            .def_readwrite("images_multi", &Output::images_multi)
            .def_readwrite("thickness_multi", &Output::thickness_multi)
            .def_readwrite("output_pts", &Output::output_pts)
            .def_readwrite("laser_rays", &Output::laser_rays)
            .def_readwrite("angles", &Output::angles)
            .def_readwrite("velocities", &Output::velocities)
            .def_readwrite("accels", &Output::accels)
            .def_readwrite("spline", &Output::spline)
            .def_readwrite("output_pts_set", &Output::output_pts_set)
            .def_readwrite("spline_set", &Output::spline_set)
            .def_readwrite("full_cloud_eig", &Output::full_cloud_eig)
            #ifdef ROS
            .def_readwrite("full_cloud", &Output::full_cloud)
            #endif
            ;

    // Spline
    py::class_<Angles, std::shared_ptr<Angles>>(m, "Angles")
            .def(py::init<>())
            .def_readwrite("max_velo", &Angles::max_velo)
            .def_readwrite("summed_peak", &Angles::summed_peak)
            .def_readwrite("design_pts", &Angles::design_pts)
            .def_readwrite("accels", &Angles::accels)
            .def_readwrite("velocities", &Angles::velocities)
            .def_readwrite("peaks", &Angles::peaks)
            ;

    // Spline
    py::class_<SplineParams>(m, "SplineParams")
            .def(py::init<>())
            ;
    py::class_<SplineParamsVec, std::shared_ptr<SplineParamsVec>>(m, "SplineParamsVec")
            .def(py::init<>())
            .def_readwrite("splineParams", &SplineParamsVec::splineParams)
            ;

    // DatumProcessor
    py::class_<DatumProcessor, std::shared_ptr<DatumProcessor>>(m, "DatumProcessor")
            .def(py::init<>())
            .def("setSensors", &DatumProcessor::setSensors)
            .def_readwrite("warnings", &DatumProcessor::warnings)
            ;

    // Fitting
    py::class_<Fitting, std::shared_ptr<Fitting>>(m, "Fitting")
            .def(py::init<std::shared_ptr<DatumProcessor>>())
            .def("curtainSplitting", &Fitting::curtainSplitting)
            .def("curtainNodes", &Fitting::curtainNodes)
            .def("splineToAngles", &Fitting::splineToAngles)
            ;

    // Planner.
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
            .def_readonly("x", &Node::x)
            .def_readonly("z", &Node::z)
            .def_readonly("r", &Node::r)
            .def_readonly("theta_cam", &Node::theta_cam)
            .def_readonly("theta_las", &Node::theta_las)
            .def_readonly("ki", &Node::ki)
            .def_readonly("kj", &Node::kj)
            ;

    py::class_<Interpolator, std::shared_ptr<Interpolator>>(m, "Interpolator");

    py::class_<CartesianNNInterpolator, Interpolator, std::shared_ptr<CartesianNNInterpolator>>(m, "CartesianNNInterpolator")
            .def(py::init<int, int, float, float, float, float> ())
            ;

    py::class_<PolarIdentityInterpolator, Interpolator, std::shared_ptr<PolarIdentityInterpolator>>(m, "PolarIdentityInterpolator")
            .def(py::init<int, int> ())
            ;

    py::class_<Planner<true>, std::shared_ptr<Planner<true>>>(m, "PlannerMax")
            .def(py::init<std::shared_ptr<DatumProcessor>, const std::vector<float>&, std::shared_ptr<Interpolator>, bool> ())
            .def("getGraphForVis", &Planner<true>::getVectorizedGraph)
            .def("optimizedDesignPts", &Planner<true>::optimizedDesignPts)
            ;

    py::class_<Planner<false>, std::shared_ptr<Planner<false>>>(m, "PlannerMin")
            .def(py::init<std::shared_ptr<DatumProcessor>, const std::vector<float>&, std::shared_ptr<Interpolator>, bool> ())
            .def("getGraphForVis", &Planner<false>::getVectorizedGraph)
            .def("optimizedDesignPts", &Planner<false>::optimizedDesignPts)
            ;

    m.def("processPointsJoint", &processPointsJoint, "processPointsJoint");
    m.def("fitBSpline", &Algo::fitBSpline, "fitBSpline");
    m.def("solveT", &Algo::solveT, "solveT");
    m.def("generateCameraAngles", &Algo::generateCameraAngles, "generateCameraAngles");
    m.def("transformPoints", &Depth::transformPoints, "transformPoints");

    #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
    #else
    m.attr("__version__") = "dev";
    #endif
}

#endif