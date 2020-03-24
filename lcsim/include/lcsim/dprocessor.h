#ifndef DPROCESS_H
#define DPROCESS_H

#include <string>
#include <iostream>
#include <common.h>

namespace lc{

class DatumProcessor{
private:
    bool set = false;
    DatumVector c_datums_, l_datums_;
    std::map<std::string, int> cam_mapping_;

public:
    DatumProcessor();
    ~DatumProcessor();

	const std::shared_ptr<Datum> getCDatum(std::string camera_name);
	std::vector<cv::Point2f> getImageCoordinates(int imgh, int imgw);
	std::vector<cv::Point2f> getImageCoordinatesXOffset(int imgh, int imgw);

	void createNormalMap(std::shared_ptr<Datum>& datum);
	Eigen::Vector4f createPlaneFromPoints(const Eigen::Matrix3f& _pts);
	Laser computeLaserParams(float t_max, Eigen::MatrixXf cam_to_laser, const Datum* l_datum);
	void setSensors(DatumVector& c_datums,  DatumVector& l_datums);

	std::vector<int> checkPoints(const std::vector<Point2D>& pts_, const Datum& cam_data, const Laser& laser_data, bool good=true);
	Eigen::Matrix4Xf findCameraIntersections(const Datum& cam_data, const std::vector<int>& good_inds_temp, const std::vector<Point2D>& pts);
	Eigen::Matrix4Xf findCameraIntersectionsOpt(const Datum& cam_data, const std::vector<int>& good_inds_temp, const std::vector<Point2D>& pts);
	std::shared_ptr<Angles> calculateAngles(const Eigen::Matrix4Xf& design_pts, const Datum& cam_data, const Laser& laser_data, bool get_pts=true, bool warn=false, bool vlimit=true);
	std::pair<cv::Mat, cv::Mat> calculateSurface(const Eigen::Matrix4Xf& design_pts, const Datum& cam_data, const Laser& laser_data);
    void computeDepthHits(std::pair<cv::Mat,cv::Mat>& surface_data, const cv::Mat& depth_img, const Datum& cam_data);

    void processPointsT(const Eigen::MatrixXf& input_pts, const cv::Mat& depth_img, std::string cam_name, std::string laser_name, cv::Mat& image, pcl::PointCloud<pcl::PointXYZRGB>& cloud, bool compute_cloud=true);

};

}

#endif