#ifndef FITTING_H
#define FITTING_H

#include <string>
#include <iostream>
#include <common.h>
#include <dprocessor.h>
#include <algo.h>

namespace lc{

class Fitting{
private:
    std::shared_ptr<DatumProcessor> datumProcessor_;

public:

    Fitting(std::shared_ptr<DatumProcessor> datumProcessor);
    ~Fitting();

    std::vector<std::vector<Point2D>> segmentClusters(const std::vector<Point2D>& design_pts, const std::vector<Point2D>& pts);
    std::vector<std::vector<Point2D>> listSort(const std::vector<std::vector<Point2D>>& list);
    std::vector<Eigen::MatrixXf> curtainSplitting(Eigen::MatrixXf& spline, std::string cam_name, std::string laser_name);

    std::shared_ptr<Angles> splineToAngles(Eigen::MatrixXf& spline, std::string cam_name, std::string laser_name, bool vlimit=true);
    std::pair<Eigen::MatrixXf, float> fitSpline(Eigen::MatrixXf& path, std::string cam_name, std::string laser_name);
    void curtainNodes(Eigen::MatrixXf& path, std::string cam_name, std::string laser_name, std::shared_ptr<Output>& output, bool process=false);

};

}

#endif