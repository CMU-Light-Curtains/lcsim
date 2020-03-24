#ifndef COMMON_H
#define COMMON_H

#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <chrono>
#include <memory>

namespace lc{

typedef Eigen::Vector2f Point2D;
typedef Eigen::Hyperplane<float,2> Line;
typedef Eigen::ParametrizedLine<float,2> Ray;
typedef cv::Vec4f PointXYZI;

class Laser{
public:
    Eigen::MatrixXf cam_to_laser, laser_to_cam;
    Eigen::Vector2f laser_origin, p_left_laser, p_right_laser;

    // Laser Params
    float galvo_m;
    float galvo_b;
    int16_t maxADC;
    float thickness;
    float divergence;
    float laser_limit;
    float laser_timestep;

    float getPositionFromAngle(float proj_angle_) const
    {
        float galvo_pos = (proj_angle_ - galvo_b)/galvo_m;
        return galvo_pos;
    }

    float getAngleFromPosition(float pos_) const
    {
        float ang = pos_*galvo_m + galvo_b;
        return ang;
    }
};

class Input{
public:
    std::string camera_name;
    //sensor_msgs::Image ros_depth_image;
    //sensor_msgs::Image ros_rgb_image;
    Eigen::MatrixXf design_pts;
    std::vector<Eigen::MatrixXf> design_pts_multi;
    Eigen::MatrixXf design_pts_conv;
    Eigen::MatrixXf surface_pts;

    cv::Mat rgb_image;
    cv::Mat depth_image;

    //sensor_msgs::PointCloud2 cloud;
    //sensor_msgs::Image image;
};

class Output{
public:
    //std::vector<sensor_msgs::PointCloud2> clouds;
    //std::vector<sensor_msgs::Image> images;
    //std::vector<std::vector<sensor_msgs::Image>> images_multi;
    //sensor_msgs::PointCloud2 full_cloud;
    Eigen::MatrixXf full_cloud_eig;

    Eigen::MatrixXf output_pts, laser_rays, spline;
    std::vector<float> angles;
    std::vector<float> velocities;
    std::vector<float> accels;
    std::vector<Eigen::MatrixXf> output_pts_set;
    std::vector<Eigen::MatrixXf> spline_set;

    //std::vector<SplineParams> all_params;
};

class Datum{
public:
    // Laser Params
    float galvo_m = -2.2450289e+01;
    float galvo_b = -6.8641598e-01;
    int16_t maxADC = 15000;
    float thickness = 0.00055;
    float divergence = 0.11/2.;
    float laser_limit = 14000;
    float laser_timestep = 1.5e-5;

    // Camera Params
    std::string type;
    std::string camera_name;
    std::string laser_name;
    //cv::Mat rgb_image;
    Eigen::MatrixXf rgb_matrix;
    //cv::Mat depth_image;
    Eigen::MatrixXf depth_matrix;
    Eigen::MatrixXf world_to_rgb;
    Eigen::MatrixXf world_to_depth;
    std::map<std::string, Eigen::MatrixXf> cam_to_laser;
    Eigen::MatrixXf cam_to_world;
    float fov;
    cv::Mat distortion;
    int imgh, imgw;
    float limit;

    // Pre Calculated
    float t_max;
    cv::Mat nmap, nmap_nn, nmap_nn_xoffset, ztoramap;
    Eigen::MatrixX3f nmap_matrix;
    cv::Mat midrays[3];
    Eigen::Vector2f cam_origin;
    Eigen::Vector2f p_left_cam, p_right_cam;
    std::map<std::string, Laser> laser_data;
    std::vector<float> valid_angles;
    //Eigen::MatrixXf design_pts_conv;
};

struct Angles{
    std::vector<float> angles;
    std::vector<float> velocities;
    std::vector<float> accels;
    std::vector<float> peaks;
    float max_velo;
    float summed_peak;
    Eigen::MatrixXf design_pts;
    Eigen::MatrixXf output_pts;
    bool exceed = false;
};


typedef std::vector<std::shared_ptr<Datum>> DatumVector;

}

#endif