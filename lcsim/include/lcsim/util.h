#ifndef UTILITY_H
#define UTILITY_H

#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#include <chrono>
#include <memory>

namespace lc{

class Util{
public:
    static Eigen::Matrix3f setEulerYPR(float eulerZ, float eulerY, float eulerX);
    static Eigen::Matrix4f getTransformMatrix(float yaw, float pitch, float roll, float x, float y, float z);
    static void intersect(float A, float B, float C, float D, const cv::Vec3f& rayEq, cv::Vec4f& coord3D);

    static std::string type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }
};



}

#endif