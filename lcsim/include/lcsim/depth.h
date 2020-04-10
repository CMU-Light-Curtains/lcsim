#ifndef DEPTH_H
#define DEPTH_H

#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#include <chrono>
#include <memory>
#include <opencv2/core/eigen.hpp>

namespace lc{

class Depth{
public:
    static Eigen::MatrixXf upsampleLidar(const Eigen::MatrixXf& lidardata_cam, std::map<std::string, float>& params);

    static Eigen::MatrixXf generateDepth(const Eigen::MatrixXf& lidardata, const Eigen::MatrixXf& intr_raw,
                                   const Eigen::MatrixXf& M_lidar2cam, int width, int height,
                                   std::map<std::string, float>& params);

    static std::vector<cv::Mat> transformPoints(const Eigen::MatrixXf& lidardata, const Eigen::MatrixXf& intr_raw,
                                           const Eigen::MatrixXf& M_lidar2cam, int width, int height,
                                           std::map<std::string, float>& params);
};



}

#endif