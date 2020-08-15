#include "util.h"

using namespace lc;

Eigen::Matrix3f Util::setEulerYPR(float eulerZ, float eulerY, float eulerX) {
    float ci = std::cos(eulerX);
    float cj = std::cos(eulerY);
    float ch = std::cos(eulerZ);
    float si = std::sin(eulerX);
    float sj = std::sin(eulerY);
    float sh = std::sin(eulerZ);
    float cc = ci * ch;
    float cs = ci * sh;
    float sc = si * ch;
    float ss = si * sh;

    Eigen::Matrix3f rot_matrix;
    rot_matrix << cj * ch, sj * sc - cs, sj * cc + ss,
            cj * sh, sj * ss + cc, sj * cs - sc,
            -sj,     cj * si,      cj * ci;

    return rot_matrix;
}

Eigen::Matrix4f Util::getTransformMatrix(float yaw, float pitch, float roll, float x, float y, float z){
    Eigen::Matrix4f transform_matrix;
    Eigen::Matrix3f rot_matrix = setEulerYPR(roll*M_PI/180., pitch*M_PI/180., yaw*M_PI/180.);
    transform_matrix(0,0) = rot_matrix(0, 0);
    transform_matrix(0,1) = rot_matrix(0, 1);
    transform_matrix(0,2) = rot_matrix(0, 2);
    transform_matrix(1,0) = rot_matrix(1, 0);
    transform_matrix(1,1) = rot_matrix(1, 1);
    transform_matrix(1,2) = rot_matrix(1, 2);
    transform_matrix(2,0) = rot_matrix(2, 0);
    transform_matrix(2,1) = rot_matrix(2, 1);
    transform_matrix(2,2) = rot_matrix(2, 2);
    transform_matrix(0,3) = x;
    transform_matrix(1,3) = y;
    transform_matrix(2,3) = z;
    transform_matrix(3,0) = 0.;
    transform_matrix(3,1) = 0.;
    transform_matrix(3,2) = 0.;
    transform_matrix(3,3) = 1.;
    return transform_matrix;
}

void Util::intersect(float A, float B, float C, float D, const cv::Vec3f& rayEq, cv::Vec4f& coord3D){
    float t = -D/(A*rayEq[0] + B*rayEq[1] + C*rayEq[2]);
    coord3D[0] = rayEq[0]*t;
    coord3D[1] = rayEq[1]*t;
    coord3D[2] = rayEq[2]*t;
    coord3D[3] = 0;
}