#include "dprocessor.h"

using namespace lc;

DatumProcessor::DatumProcessor(){
}

DatumProcessor::~DatumProcessor(){
}

const std::shared_ptr<Datum> DatumProcessor::getCDatum(std::string camera_name){
    return c_datums_[cam_mapping_[camera_name]];
}

std::vector<cv::Point2f> DatumProcessor::getImageCoordinates(int imgh, int imgw) {
    std::vector<cv::Point2f> img_coords;
    for (int i = 0; i < imgw; i++)
    {
        for (int j = 0; j < imgh; j++) {
            cv::Point2f pt_tmp(i+1,j+1);
            img_coords.push_back(pt_tmp);
        }
    }
    return img_coords;
}

std::vector<cv::Point2f> DatumProcessor::getImageCoordinatesXOffset(int imgh, int imgw) {
    std::vector<cv::Point2f> img_coords;
    for (int i = 0; i <= imgw; i++)
    {
        for (int j = 0; j < imgh; j++) {
            cv::Point2f pt_tmp(i+1.-0.5,j+1);
            img_coords.push_back(pt_tmp);
        }
    }
    return img_coords;
}

void DatumProcessor::createNormalMap(std::shared_ptr<Datum>& datum){
    datum->nmap_matrix = Eigen::MatrixX3f(datum->imgh*datum->imgw,3);
    datum->nmap = cv::Mat(datum->imgh, datum->imgw, CV_32FC3); //create 3 channel matrix to store rays for each pixel (x, y, z)
    datum->nmap_nn = cv::Mat(datum->imgh, datum->imgw, CV_32FC3); //create 3 channel matrix to store rays for each pixel (x, y, z)
    datum->nmap_nn_xoffset = cv::Mat(datum->imgh, datum->imgw+1, CV_32FC3); //create 3 channel matrix to store rays for each pixel (x, y, z)
    datum->ztoramap = cv::Mat(datum->imgh, datum->imgw, CV_32FC1);

    std::vector<cv::Point2f> img_coords;
    std::vector<cv::Point2f> norm_coords(datum->imgh * datum->imgw);
    img_coords = getImageCoordinates(datum->imgh, datum->imgw);

    cv::Mat KK_;//(KK().rows(), KK().cols(), CV_32FC1, KK().data());
    cv::Mat Kd_;//(Kd().rows(), Kd().cols(), CV_32FC1, Kd().data());
    cv::eigen2cv(datum->depth_matrix, KK_);
    Kd_ = datum->distortion;

    cv::undistortPoints(img_coords, norm_coords, KK_, Kd_);

    for (int i = 0; i < datum->imgw; i++)
    {
        for (int j = 0; j < datum->imgh; j++)
        {
            int pix_num = i*datum->imgh+j;
            float x_dir = norm_coords[pix_num].x;
            float y_dir = norm_coords[pix_num].y;
            float z_dir = 1.0f;
            float mag = sqrt(x_dir*x_dir + y_dir*y_dir + z_dir*z_dir);
            cv::Vec3f pix_ray = cv::Vec3f(x_dir/mag,y_dir/mag,z_dir/mag);
            cv::Vec3f pix_ray_nn = cv::Vec3f(x_dir,y_dir,z_dir);
            datum->nmap.at<cv::Vec3f>(j,i) = pix_ray;
            datum->nmap_nn.at<cv::Vec3f>(j,i) = pix_ray_nn;
            Eigen::Vector3f vector = Eigen::Vector3f(x_dir/mag,y_dir/mag,z_dir/mag);
            datum->nmap_matrix.row(j+datum->imgh*i) = vector.transpose();
            // 3d pt
            float theta = atan2(vector(0), vector(2));
            float phi = asin(vector(1)/1);
            datum->ztoramap.at<float>(j,i) = cos(theta)*cos(phi);
        }
    }

    // Split midrays into 3 channels, x(0), y(1), z(2)
    split(datum->nmap.row(datum->imgh/2-1).clone(),datum->midrays);

    // Compute Valid Angles (If we compute design points with these theta, they are guaranteed to lie on ray)
    for(int i=0; i<datum->midrays[0].size().width; i++){
        float x = datum->midrays[0].at<float>(0,i);
        float y = datum->midrays[1].at<float>(0,i);
        float z = datum->midrays[2].at<float>(0,i);
        float theta = -((atan2f(z, x) * 180 / M_PI) - 90);
        datum->valid_angles.emplace_back(theta);
    }

    // Set origin point for cx,cy as 0,0
    Eigen::Vector2f o_pt(0,0);
    datum->cam_origin = o_pt;

    // Compute the leftmost and rightmost ray that sits on xz plane (Why store at Vector2f)
    Eigen::Vector3f cam_axis(0,1,0);
    Eigen::Vector2f left_cam_ray(datum->midrays[0].at<float>(0,0),datum->midrays[2].at<float>(0,0));
    Eigen::Vector2f right_cam_ray(datum->midrays[0].at<float>(0,datum->imgw-1),datum->midrays[2].at<float>(0,datum->imgw-1));

    // Compute leftmost and rightmost point
    float z_max = 1;
    float t = 1.25*z_max/left_cam_ray(1);
    datum->p_left_cam = datum->cam_origin + t*left_cam_ray;
    datum->p_right_cam = datum->cam_origin + t*right_cam_ray;
    datum->t_max = t;

    // Offset Coords
    std::vector<cv::Point2f> img_coords_xoffset;
    std::vector<cv::Point2f> norm_coords_xoffset(datum->imgh * (datum->imgw+1));
    img_coords_xoffset = getImageCoordinatesXOffset(datum->imgh, datum->imgw);
    cv::undistortPoints(img_coords_xoffset, norm_coords_xoffset, KK_, Kd_);

    // Offset Coords rays
    for (int i = 0; i <= datum->imgw; i++)
    {
        for (int j = 0; j < datum->imgh; j++)
        {
            int pix_num = i*datum->imgh+j;
            float x_dir = norm_coords_xoffset[pix_num].x;
            float y_dir = norm_coords_xoffset[pix_num].y;
            float z_dir = 1.0f;
            cv::Vec3f pix_ray_nn_xoffset = cv::Vec3f(x_dir,y_dir,z_dir);
            datum->nmap_nn_xoffset.at<cv::Vec3f>(j,i) = pix_ray_nn_xoffset;
        }
    }

    // K'p to compute camera ray - https://nghiaho.com/?page_id=363
    // But what about the distortion param
}

Eigen::Vector4f DatumProcessor::createPlaneFromPoints(const Eigen::Matrix3f& _pts)
{
    Eigen::Vector3f P0 = _pts.row(0);
    Eigen::Vector3f P1 = _pts.row(1);
    Eigen::Vector3f P2 = _pts.row(2);

    Eigen::Vector3f P0P1 = P1-P0;
    Eigen::Vector3f P0P2 = P2-P0;

    Eigen::Vector3f n = P0P1.cross(P0P2);
    float d = P0.dot(n);

    Eigen::Vector4f plane(n(0),n(1),n(2),d);

    return plane;
}

Laser DatumProcessor::computeLaserParams(float t_max, Eigen::MatrixXf cam_to_laser, const Datum* l_datum){
    Laser laser;

    // Copy from Datum
    laser.galvo_m = l_datum->galvo_m;
    laser.galvo_b = l_datum->galvo_b;
    laser.maxADC = l_datum->maxADC;
    laser.thickness = l_datum->thickness;
    laser.divergence = l_datum->divergence;
    laser.laser_limit = l_datum->laser_limit;
    laser.laser_timestep = l_datum->laser_timestep;

    // Transforms and origin
    laser.cam_to_laser = cam_to_laser;
    laser.laser_to_cam = cam_to_laser.inverse();
    Eigen::Vector2f tmp_laser_origin(laser.laser_to_cam(0,3),laser.laser_to_cam(2,3));
    laser.laser_origin = tmp_laser_origin;

    // Hardcode laser fov angle limit
    float thetad_left = laser.getAngleFromPosition(1);
    float thetad_right = laser.getAngleFromPosition(-1);
    if(l_datum->fov > 0){
        thetad_left = -l_datum->fov/2.;
        thetad_right = l_datum->fov/2.;
    }else{
        std::cout << "Using Default Laser Params" << std::endl;
    }

    // Left points
    Eigen::Matrix3f laser_pts_left;
    laser_pts_left <<   0, 0, 0,
            0, 1, 0,
            sin(M_PI/180*thetad_left), 0, cos(M_PI/180*thetad_left);
    Eigen::Vector4f laser_plane_left_lframe = createPlaneFromPoints(laser_pts_left);
    Eigen::Vector4f laser_plane_left_cframe = laser_plane_left_lframe.transpose()*(laser.laser_to_cam.inverse());
    Eigen::Vector3f dir_tmp_l(0,-1,0);
    Eigen::Vector3f las_tmp_l = laser_plane_left_cframe.topLeftCorner(3,1);//drop the distance element from the bottom
    Eigen::Vector3f l_vec_cframe_left = dir_tmp_l.cross(las_tmp_l);
    Eigen::Vector2f lvec_tmp_l(l_vec_cframe_left(0),l_vec_cframe_left(2));
    laser.p_left_laser = laser.laser_origin + t_max*lvec_tmp_l;

    // Right points
    Eigen::Matrix3f laser_pts_right;
    laser_pts_right <<  0, 0, 0,
            0, 1, 0,
            sin(M_PI/180*thetad_right), 0, cos(M_PI/180*thetad_right);
    Eigen::Vector4f laser_plane_right_lframe = createPlaneFromPoints(laser_pts_right);
    Eigen::Vector4f laser_plane_right_cframe = laser_plane_right_lframe.transpose()*(laser.laser_to_cam.inverse());
    Eigen::Vector3f dir_tmp_r(0,-1,0);
    Eigen::Vector3f las_tmp_r = laser_plane_right_cframe.topLeftCorner(3,1); //drop the distance element from the bottom
    Eigen::Vector3f l_vec_cframe_right = dir_tmp_r.cross(las_tmp_r);
    Eigen::Vector2f lvec_tmp_r(l_vec_cframe_right(0),l_vec_cframe_right(2));
    laser.p_right_laser = laser.laser_origin + t_max*lvec_tmp_r;

    return laser;
}

void DatumProcessor::setSensors(DatumVector& c_datums,  DatumVector& l_datums){
    c_datums_ = c_datums;
    l_datums_ = l_datums;
    set = true;

    // Iterate each camera
    int i=0;
    for(auto& c_datum : c_datums_){

        // Mapping
        cam_mapping_[c_datum->camera_name] = i;
        i++;

        // Create normal map and store various rays and angles
        createNormalMap(c_datum);

        // For each laser we compute some additional params
        for(auto& l_datum : l_datums){
            c_datum->laser_data[l_datum->laser_name] = computeLaserParams(c_datum->t_max, c_datum->cam_to_laser[l_datum->laser_name], l_datum.get());
        }

    }
}

// Check if the points in cam frame lie within the two modalities
std::vector<int> DatumProcessor::checkPoints(const std::vector<Point2D>& pts_, const Datum& cam_data, const Laser& laser_data, bool good){
    std::vector<int> good_inds;
    std::vector<int> bad_inds;

    float x, z;
    float x1, z1, x2, z2;
    float d_cam_left, d_cam_right, d_laser_left, d_laser_right;

    int i = 0;
    for (auto &pt: pts_) {
        x = pt(0); z = pt(1);

        //check if point is inside left camera bound
        x1 = cam_data.cam_origin(0); z1 = cam_data.cam_origin(1);
        x2 = cam_data.p_left_cam(0); z2 = cam_data.p_left_cam(1);
        d_cam_left = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1); //d>0 if inside bound

        //check if point is inside right camera bound
        x2 = cam_data.p_right_cam(0); z2 = cam_data.p_right_cam(1);
        d_cam_right = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1); //d<0 if inside bound

        //check if point is inside left projector bound
        x1 = laser_data.laser_origin(0); z1 = laser_data.laser_origin(1);
        x2 = laser_data.p_left_laser(0); z2 = laser_data.p_left_laser(1);
        d_laser_left = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1); //d>0 if inside bound

        //check if point is inside right projector bound
        x2 = laser_data.p_right_laser(0); z2 = laser_data.p_right_laser(1);
        d_laser_right = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1); //d<0 if inside bound

        if (d_cam_left > 0 && d_laser_left > 0 && d_cam_right < 0 && d_laser_right < 0) {
            good_inds.push_back(i);
        }else{
            bad_inds.push_back(i);
        }
        i++;
    }

    if(good)
        return good_inds;
    else
        return bad_inds;
}

Eigen::Matrix4Xf DatumProcessor::findCameraIntersections(const Datum& cam_data, const std::vector<int>& good_inds_temp, const std::vector<Point2D>& pts)
{
    // ASK Question
    // What does imgh actually refer to. Since we know the camera is rotated. Should I be using imgw
    // I have changed it to imgw

    std::vector<int> good_inds = good_inds_temp;
    if(good_inds.size() == 0){
        good_inds.resize(pts.size());
        for(int i=0; i<good_inds.size(); i++) {
            good_inds[i] = i;
        }
    }

    // Empty Matrix - Number of points of size rows
    Eigen::Matrix4Xf design_pts(4, cam_data.imgw);
    int valid_points = 0;

    // Calculate camera ray intersections for design points
    Eigen::Vector2f p0(0, 0);

    // Iterate each column/ray of the camera
    //#pragma omp parallel for shared(cam_data, design_pts)
    for (int i = 0; i < cam_data.imgw; i++) {

        // Get Ray along mid
        Eigen::Vector2f dir(cam_data.midrays[0].at<float>(0, i), cam_data.midrays[2].at<float>(0, i));
        dir.normalize();
        Ray cam_ray = Ray(p0, dir);

        // Iterate the valid set of points (Need to ensure the points are continuous)
        bool found_intersection;
        Eigen::Vector2f intersection_pt;
        for (int j = 0; j < (good_inds.size() - 1); j++)
        {
            Eigen::Vector2f p1(pts[good_inds[j]](0), pts[good_inds[j]](1));
            Eigen::Vector2f p2(pts[good_inds[j + 1]](0), pts[good_inds[j + 1]](1));

            // Create the intersection point for camera
            Line p1p2 = Line::Through(p1, p2);
            Eigen::Vector2f pt = cam_ray.intersectionPoint(p1p2); //guaranteed to be on line from p1 to p2

            // Check if pt is between the two design points
            // from: https://www.lucidar.me/en/mathematics/check-if-a-point-belongs-on-a-line-segment/
            Eigen::Vector2f p1p2_vec = p2 - p1;
            Eigen::Vector2f p1pt_vec = pt - p1;

            float k_p1p2 = p1p2_vec.dot(p1p2_vec); //max distance if point is between p1 and p2;
            float k_p1pt = p1p2_vec.dot(p1pt_vec);

            if (k_p1pt < 0)
                found_intersection = false;
            else if (k_p1pt > k_p1p2)
                found_intersection = false;
            else if (abs(k_p1pt) < FLT_EPSILON) {
                found_intersection = true;
                intersection_pt = pt;
                break;
            } else if (abs(k_p1pt - k_p1p2) < FLT_EPSILON) {
                found_intersection = true;
                intersection_pt = pt;
                break;
            } else if (k_p1pt > 0 && k_p1pt < k_p1p2) {
                found_intersection = true;
                intersection_pt = pt;
                break;
            }
            else
                found_intersection = false;
        }

        float cp = (float)i/float(cam_data.imgw);
        if(cam_data.limit > 0) if(cp > cam_data.limit) found_intersection = false;
        if(cam_data.limit < 0) if(cp < fabs(cam_data.limit)) found_intersection = false;

        if (found_intersection) {
            design_pts(0, i) = intersection_pt(0); //x-value of pt
            design_pts(1, i) = 0; //y-value of pt (zero since in xz plane)
            design_pts(2, i) = intersection_pt(1); //z-value of pt
            design_pts(3, i) = 1; // 1 to make pt homogenous
            valid_points+=1;
        } else {
            design_pts(0, i) = 0; //x-value of pt
            design_pts(1, i) = 0; //y-value of pt (zero since in xz plane)
            design_pts(2, i) = 0; //z-value of pt
            design_pts(3, i) = -1; // -1 indicates bad point
        }

    }

    return design_pts;
}

//Eigen::Matrix4Xf DatumProcessor::findCameraIntersectionsOpt(const Datum& cam_data, const std::vector<int>& good_inds_temp, const std::vector<Point2D>& pts)
//{
//    std::vector<int> good_inds = good_inds_temp;
//    if(good_inds.size() == 0){
//        good_inds.resize(pts.size());
//        for(int i=0; i<good_inds.size(); i++) {
//            good_inds[i] = i;
//        }
//    }
//
//    // Empty Matrix - Number of points of size rows
//    Eigen::Matrix4Xf design_pts(4, cam_data.imgw);
//    int valid_points = 0;
//
//    // Calculate camera ray intersections for design points
//    Eigen::Vector2f p0(0, 0);
//
//    // Bins
//    std::vector<std::vector<int>> bins;
//    bins.resize(cam_data.imgw + 1);
//
//    // Iterate through each angle
//    for(int i=0; i<pts.size(); i++){
//        if(std::find(good_inds.begin(), good_inds.end(), i) == good_inds.end()) continue;
//        float pt_angle = -((atan2f(pts[i](1), pts[i](0)) * 180 / M_PI) - 90) + 0 + 0;
//
//        // Iterate through each bin to find best bin
//        for(int j=0; j<bins.size(); j++) {
//            float left_angle;
//            float right_angle;
//            if(j == 0){
//                left_angle = -100;
//                right_angle = cam_data.valid_angles[j];
//            }else if(j == bins.size()-1){
//                left_angle = cam_data.valid_angles[j-1];
//                right_angle = 100;
//            }else{
//                left_angle = cam_data.valid_angles[j-1];
//                right_angle = cam_data.valid_angles[j];
//            }
//
//            // Add if satisfies
//            if (pt_angle > left_angle && pt_angle < right_angle)  {
//                bins[j].emplace_back(i);
//                break;
//            }
//        }
//
//        // Sort by range?
//    }
//
//
//    // Idea for Opt 2 for interpolate just try again
//    for (int i = 0; i < cam_data.imgw; i++) {
//
//        // Get Ray along mid
//        Eigen::Vector2f dir(cam_data.midrays[0].at<float>(0, i), cam_data.midrays[2].at<float>(0, i));
//        dir.normalize();
//        Ray cam_ray = Ray(p0, dir);
//        float cam_angle = cam_data.valid_angles[i];
//
//        // Iterate the valid set of points (Need to ensure the points are continuous)
//        bool found_intersection = false;
//        Eigen::Vector2f intersection_pt(0.,0.);
//
//        // Start at right bin and search for points
//        int g1 = -1;
//        int g2 = -1;
//        for(int j=i; j>=0; j--){
//            if(!bins[j].size()) continue;
//            g1 = bins[j].front(); // need a way to pop out?
//            // Keep searching bin until i find one that is close?
//            bins[j].erase(bins[j].begin());
//            break;
//        }
//        for(int j=i+1; j<bins.size(); j++){
//            if(!bins[j].size()) continue;
//            g2 = bins[j].front();
//            //bins[j].erase(bins[j].begin());
//            break;
//        }
//
//        if(g1 != -1 && g2 != -1){
//            Eigen::Vector2f p1(pts[g1](0), pts[g1](1));
//            Eigen::Vector2f p2(pts[g2](0), pts[g2](1));
//
//            // Create the intersection point for camera
//            Line p1p2 = Line::Through(p1, p2);
//            Eigen::Vector2f pt = cam_ray.intersectionPoint(p1p2); //guaranteed to be on line from p1 to p2
//
//            // Check if pt is between the two design points
//            // from: https://www.lucidar.me/en/mathematics/check-if-a-point-belongs-on-a-line-segment/
//            Eigen::Vector2f p1p2_vec = p2 - p1;
//            Eigen::Vector2f p1pt_vec = pt - p1;
//
//            float k_p1p2 = p1p2_vec.dot(p1p2_vec); //max distance if point is between p1 and p2;
//            float k_p1pt = p1p2_vec.dot(p1pt_vec);
//
//            if (k_p1pt < 0)
//                found_intersection = false;
//            else if (k_p1pt > k_p1p2)
//                found_intersection = false;
//            else if (abs(k_p1pt) < FLT_EPSILON) {
//                found_intersection = true;
//                intersection_pt = pt;
//            } else if (abs(k_p1pt - k_p1p2) < FLT_EPSILON) {
//                found_intersection = true;
//                intersection_pt = pt;
//            } else if (k_p1pt > 0 && k_p1pt < k_p1p2) {
//                found_intersection = true;
//                intersection_pt = pt;
//            }
//            else
//                found_intersection = false;
//        }
//
//        float cp = (float)i/float(cam_data.imgw);
//        if(cam_data.limit > 0) if(cp > cam_data.limit) found_intersection = false;
//        if(cam_data.limit < 0) if(cp < fabs(cam_data.limit)) found_intersection = false;
//
//        if (found_intersection) {
//            design_pts(0, i) = intersection_pt(0); //x-value of pt
//            design_pts(1, i) = 0; //y-value of pt (zero since in xz plane)
//            design_pts(2, i) = intersection_pt(1); //z-value of pt
//            design_pts(3, i) = 1; // 1 to make pt homogenous
//            valid_points+=1;
//        } else {
//            design_pts(0, i) = 0; //x-value of pt
//            design_pts(1, i) = 0; //y-value of pt (zero since in xz plane)
//            design_pts(2, i) = 0; //z-value of pt
//            design_pts(3, i) = -1; // -1 indicates bad point
//        }
//
//    }
//
//    return design_pts;
//
//}

Eigen::Matrix4Xf DatumProcessor::findCameraIntersectionsOpt(const Datum& cam_data, const std::vector<int>& good_inds_temp, const std::vector<Point2D>& pts)
{
    // This function only optimal at continuous regions
    std::vector<int> good_inds = good_inds_temp;
    if(good_inds.size() == 0){
        good_inds.resize(pts.size());
        for(int i=0; i<good_inds.size(); i++) {
            good_inds[i] = i;
        }
    }

    // Empty Matrix - Number of points of size rows
    Eigen::Matrix4Xf design_pts(4, cam_data.imgw);
    int valid_points = 0;

    // Calculate camera ray intersections for design points
    Eigen::Vector2f p0(0, 0);

    // Store angles
    float nanVal = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> angles;
    angles.resize(pts.size());
    for(int i=0; i<angles.size(); i++){
        angles[i] = -((atan2f(pts[i](1), pts[i](0)) * 180 / M_PI) - 90) + 0 + 0;
    }

    // Iterate each column/ray of the camera
    //#pragma omp parallel for shared(cam_data, design_pts)
    for (int i = 0; i < cam_data.imgw; i++) {

        // Get Ray along mid
        Eigen::Vector2f dir(cam_data.midrays[0].at<float>(0, i), cam_data.midrays[2].at<float>(0, i));
        dir.normalize();
        Ray cam_ray = Ray(p0, dir);
        float cam_angle = cam_data.valid_angles[i];

        // Iterate the valid set of points (Need to ensure the points are continuous)
        std::vector<float> limits = {1, 100};
        bool found_intersection = false;
        Eigen::Vector2f intersection_pt;
        for(auto limit : limits)
        {
            found_intersection = false;
            for (int j = 0; j < (good_inds.size() - 1); j++)
            {
                int p1index = good_inds[j];
                int p2index = good_inds[j+1];

                float p1angle = angles[p1index];
                float p2angle = angles[p2index];

                if(p1angle != p2angle)
                {
                    //float limit = 100;
                    if(fabs(p1angle - cam_angle) > limit && fabs(p2angle - cam_angle) > limit) continue;
                }

                Eigen::Vector2f p1(pts[p1index](0), pts[p1index](1));
                Eigen::Vector2f p2(pts[p2index](0), pts[p2index](1));
                if(p1 == p2) continue;

                // Create the intersection point for camera
                Line p1p2 = Line::Through(p1, p2);
                Eigen::Vector2f pt = cam_ray.intersectionPoint(p1p2); //guaranteed to be on line from p1 to p2

                // Check if pt is between the two design points
                // from: https://www.lucidar.me/en/mathematics/check-if-a-point-belongs-on-a-line-segment/
                Eigen::Vector2f p1p2_vec = p2 - p1;
                Eigen::Vector2f p1pt_vec = pt - p1;

                float k_p1p2 = p1p2_vec.dot(p1p2_vec); //max distance if point is between p1 and p2;
                float k_p1pt = p1p2_vec.dot(p1pt_vec);

                if (k_p1pt < 0)
                    found_intersection = false;
                else if (k_p1pt > k_p1p2)
                    found_intersection = false;
                else if (abs(k_p1pt) < FLT_EPSILON) {
                    found_intersection = true;
                    intersection_pt = pt;
                } else if (abs(k_p1pt - k_p1p2) < FLT_EPSILON) {
                    found_intersection = true;
                    intersection_pt = pt;
                } else if (k_p1pt > 0 && k_p1pt < k_p1p2) {
                    found_intersection = true;
                    intersection_pt = pt;
                }
                else
                    found_intersection = false;

                if(found_intersection){
                    //if(p1angle != p2angle)
                    //    std::cout << cam_angle << " " << p1angle << " " << p2angle << " " << p1index << " " << p2index <<  std::endl;
                    break;
                }
            }

            if(found_intersection) break;
        }

        float cp = (float)i/float(cam_data.imgw);
        if(cam_data.limit > 0) if(cp > cam_data.limit) found_intersection = false;
        if(cam_data.limit < 0) if(cp < fabs(cam_data.limit)) found_intersection = false;

        if (found_intersection) {
            design_pts(0, i) = intersection_pt(0); //x-value of pt
            design_pts(1, i) = 0; //y-value of pt (zero since in xz plane)
            design_pts(2, i) = intersection_pt(1); //z-value of pt
            design_pts(3, i) = 1; // 1 to make pt homogenous
            valid_points+=1;
        } else {
            design_pts(0, i) = 0; //x-value of pt
            design_pts(1, i) = 0; //y-value of pt (zero since in xz plane)
            design_pts(2, i) = 0; //z-value of pt
            design_pts(3, i) = -1; // -1 indicates bad point
        }

    }

    return design_pts;
}

std::shared_ptr<Angles> DatumProcessor::calculateAngles(const Eigen::Matrix4Xf& design_pts, const Datum& cam_data, const Laser& laser_data, bool get_pts, bool warn, bool vlimit){
    std::shared_ptr<Angles> angles_ptr = std::make_shared<Angles>();
    Angles& angles = *angles_ptr.get();
    Eigen::Matrix4Xf design_pts_laser = laser_data.cam_to_laser * design_pts;

    // Calculate Galvo angle check
    float nanVal = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> laser_angles;
    laser_angles.resize(design_pts_laser.cols());
    for(int i=0; i<design_pts_laser.cols(); i++){
        if(design_pts_laser(3, i) == -1){
            laser_angles[i] = nanVal;
            continue;
        }
        laser_angles[i] = -((atan2f(design_pts_laser(2, i), design_pts_laser(0, i)) * 180 / M_PI) - 90) + 0 + 0;
    }

    // Smooth out angle?
    bool exceed = false;
    std::vector<float> laser_angles_temp = {laser_angles[0]};
    std::vector<float> velocities = {0};
    for(int i=1; i<laser_angles.size(); i++){
        auto new_pt = laser_angles[i];
        auto old_pt = laser_angles_temp.back();
        auto velo = (new_pt - old_pt)/(laser_data.laser_timestep);
        velocities.emplace_back(velo);
        if(fabs(velo) > laser_data.laser_limit && vlimit){
            new_pt = old_pt + laser_data.laser_limit*laser_data.laser_timestep*copysignf(1.0, velo);
            exceed = true;
        }
        laser_angles_temp.emplace_back(new_pt);
    }
    laser_angles = laser_angles_temp;
    velocities[0] = velocities[1];
    std::vector<float> accel = {0};
    for(int i=1; i<velocities.size(); i++){
        auto new_pt = velocities[i];
        auto old_pt = velocities[i-1];
        accel.emplace_back(old_pt - new_pt);
    }
    accel[0] = accel[1];
    angles.angles = laser_angles;
    angles.velocities = velocities;
    angles.accels = accel;
    if(exceed) angles.exceed = true;
    if(exceed) if(warn) std::cerr << "Design points have exceeded laser limit" << std::endl;

    if(!get_pts) return angles_ptr;

    Eigen::Matrix4Xf planes_lframe = Eigen::Matrix4Xf::Zero(4, laser_angles.size());
    for(int i=0; i<laser_angles.size(); i++){
        auto laser_angle = laser_angles[i];
        //if(!std::isnan(laser_angles[i])) std::cout << laser_angle << std::endl;

        // Create a straight plane
        Eigen::Matrix4Xf straight_plane(4,1);
        straight_plane(0, 0) = 1.; // should we flip this?
        straight_plane(1, 0) = 0.;
        straight_plane(2, 0) = 0.;
        straight_plane(3, 0) = 0.;

        // Rotation Matrices
        Eigen::Matrix4f lrotated_matrix = Util::getTransformMatrix(0,laser_angle,0,0,0,0);
        Eigen::Matrix4Xf lrotated_plane = ((lrotated_matrix.inverse().transpose())*straight_plane);
        planes_lframe.col(i) = lrotated_plane;
    }

    // Transform planes to camera frame
    // https://math.stackexchange.com/questions/1377107/new-plane-equation-after-transformation-of-coordinates
    Eigen::Matrix4Xf planes_cframe = ((laser_data.laser_to_cam.inverse().transpose())*planes_lframe);

    // Compute Ray Intersection
    // https://nghiaho.com/?page_id=363
    float* plane_data = planes_cframe.data();
    float pixelrows = planes_cframe.cols();
    Eigen::Matrix4Xf design_pts_new = design_pts;
    for(int u=0; u<pixelrows; u++){
        float v = cam_data.imgh/2.- 1;

        // Intersect
        float A = plane_data[u*4 + 0];
        float B = plane_data[u*4 + 1];
        float C = plane_data[u*4 + 2];
        float D = plane_data[u*4 + 3];
        cv::Vec4f design_pt;
        Util::intersect(A, B, C, D, cam_data.nmap_nn.at<cv::Vec3f>(v,u), design_pt);

        // Store
        design_pts_new(0,u) = design_pt[0];
        design_pts_new(1,u) = design_pt[1];
        design_pts_new(2,u) = design_pt[2];
        design_pts_new(3,u) = design_pt[3];
    }

    angles.output_pts = design_pts_new;

    return angles_ptr;
}

std::pair<cv::Mat, cv::Mat> DatumProcessor::calculateSurface(const Eigen::Matrix4Xf& design_pts, const Datum& cam_data, const Laser& laser_data){
    auto start = std::chrono::steady_clock::now();

    std::pair<cv::Mat, cv::Mat> surface_data;
    cv::Mat& surface_pts = surface_data.first;
    cv::Mat& surface_unc = surface_data.second;

    surface_pts = cv::Mat(cam_data.nmap_nn.size().height, cam_data.nmap_nn.size().width, CV_32FC4);

    // Params
    float nanVal = std::numeric_limits<float>::quiet_NaN();
    int numCols = design_pts.cols();
    int numRows = cam_data.nmap.rows;
    if(cam_data.nmap.cols != numCols)
        throw std::runtime_error("Nmap and design pts dont match");

    // Store the points in laser (-1 in last column if invalid)
    Eigen::Matrix4Xf design_pts_laser = laser_data.cam_to_laser * design_pts;

    // Calculate Galvo angle check
    std::vector<float> laser_angles;
    laser_angles.resize(design_pts_laser.cols());
    for(int i=0; i<design_pts_laser.cols(); i++){
        if(design_pts_laser(3, i) == -1){
            laser_angles[i] = nanVal;
            continue;
        }
        laser_angles[i] = -((atan2f(design_pts_laser(2, i), design_pts_laser(0, i)) * 180 / M_PI) - 90) + 0 + 0;
    }

    // Smooth out angle?
    bool exceed = false;
    std::vector<float> laser_angles_temp = {laser_angles[0]};
    for(int i=1; i<laser_angles.size(); i++){
        auto new_pt = laser_angles[i];
        auto old_pt = laser_angles_temp.back();
        auto velo = (new_pt - old_pt)/(laser_data.laser_timestep);

        if(fabs(velo) > laser_data.laser_limit){
            new_pt = old_pt + laser_data.laser_limit*laser_data.laser_timestep*copysignf(1.0, velo);
            exceed = true;
        }
        laser_angles_temp.emplace_back(new_pt);
    }
    //laser_angles = laser_angles_temp;
    //if(exceed) ROS_WARN("Design points have exceeded laser limit");

    Eigen::Matrix4Xf planes_lframe = Eigen::Matrix4Xf::Zero(4, laser_angles.size());
    for(int i=0; i<laser_angles.size(); i++){
        auto laser_angle = laser_angles[i];
        //if(!std::isnan(laser_angles[i])) std::cout << laser_angle << std::endl;

        // Create a straight plane
        Eigen::Matrix4Xf straight_plane(4,1);
        straight_plane(0, 0) = 1.; // should we flip this?
        straight_plane(1, 0) = 0.;
        straight_plane(2, 0) = 0.;
        straight_plane(3, 0) = 0.;

        // Rotation Matrices
        Eigen::Matrix4f lrotated_matrix = Util::getTransformMatrix(0,laser_angle,0,0,0,0);
        Eigen::Matrix4Xf lrotated_plane = ((lrotated_matrix.inverse().transpose())*straight_plane);
        planes_lframe.col(i) = lrotated_plane;
    }

    // Transform planes to camera frame
    // https://math.stackexchange.com/questions/1377107/new-plane-equation-after-transformation-of-coordinates
    Eigen::Matrix4Xf planes_cframe = ((laser_data.laser_to_cam.inverse().transpose())*planes_lframe);

    // Compute Ray Intersection
    // https://nghiaho.com/?page_id=363
    float* plane_data = planes_cframe.data();
    for(int v=0; v<surface_pts.size().height; v++){
        for(int u=0; u<surface_pts.size().width; u++){
            // Got to handle invalid angles here

            // Intersect
            float A = plane_data[u*4 + 0];
            float B = plane_data[u*4 + 1];
            float C = plane_data[u*4 + 2];
            float D = plane_data[u*4 + 3];
            Util::intersect(A, B, C, D, cam_data.nmap_nn.at<cv::Vec3f>(v,u), surface_pts.at<cv::Vec4f>(v,u));

            if(surface_pts.at<cv::Vec4f>(v,u)[2] < 0 || fabs(surface_pts.at<cv::Vec4f>(v,u)[1]) > 3.0){
                surface_pts.at<cv::Vec4f>(v,u)[0] = nanVal;
                surface_pts.at<cv::Vec4f>(v,u)[1] = nanVal;
                surface_pts.at<cv::Vec4f>(v,u)[2] = nanVal;
                surface_pts.at<cv::Vec4f>(v,u)[3] = nanVal;
            }
        }
    }

    // Setup Unc
    surface_unc = cv::Mat(cam_data.imgh, cam_data.imgw, CV_32FC1);

    // Compute the various planes for divergence and thickness
    Eigen::Matrix4Xf lrotated_planes_lframe = Eigen::Matrix4Xf::Zero(4, laser_angles.size());
    Eigen::Matrix4Xf rrotated_planes_lframe = Eigen::Matrix4Xf::Zero(4, laser_angles.size());
    for(int i=0; i<laser_angles.size(); i++){
        if(std::isnan(laser_angles[i])) continue;
        float laser_angle = laser_angles[i];

        // Create a straight plane
        Eigen::Matrix4Xf straight_plane(4,1);
        straight_plane(0, 0) = 1.;
        straight_plane(1, 0) = 0.;
        straight_plane(2, 0) = 0.;
        straight_plane(3, 0) = 0.;

        // Rotation Matrices
        Eigen::Matrix4f lrotated_matrix = Util::getTransformMatrix(0,laser_angle,0,0,0,0) * Util::getTransformMatrix(0,0,0,(laser_data.thickness/2),0,0) * Util::getTransformMatrix(0,-laser_data.divergence,0,0,0,0);
        Eigen::Matrix4f rrotated_matrix = Util::getTransformMatrix(0,laser_angle,0,0,0,0) * Util::getTransformMatrix(0,0,0,-(laser_data.thickness/2),0,0) * Util::getTransformMatrix(0,laser_data.divergence,0,0,0,0);

        // Transform planes
        Eigen::Matrix4Xf lrotated_plane = ((lrotated_matrix.inverse().transpose())*straight_plane);
        Eigen::Matrix4Xf rrotated_plane = ((rrotated_matrix.inverse().transpose())*straight_plane);

        // Set it
        lrotated_planes_lframe.col(i) = lrotated_plane;
        rrotated_planes_lframe.col(i) = rrotated_plane;
    }

    // Transform to Cam
    Eigen::Matrix4Xf lrotated_planes_cframe = ((laser_data.laser_to_cam.inverse().transpose())*lrotated_planes_lframe);
    Eigen::Matrix4Xf rrotated_planes_cframe = ((laser_data.laser_to_cam.inverse().transpose())*rrotated_planes_lframe);

    // Now iterate
    float* left_plane_data = lrotated_planes_cframe.data();
    float* right_plane_data = rrotated_planes_cframe.data();
    for(int v=0; v<surface_pts.size().height; v++){
        for(int u=0; u<surface_pts.size().width; u++){
            // Got to handle invalid angles here
            // if(std::isnan(laser_angles[u])) continue;

            // Planes
            float Al = left_plane_data[u*4 + 0];
            float Bl = left_plane_data[u*4 + 1];
            float Cl = left_plane_data[u*4 + 2];
            float Dl = left_plane_data[u*4 + 3];
            float Ar = right_plane_data[u*4 + 0];
            float Br = right_plane_data[u*4 + 1];
            float Cr = right_plane_data[u*4 + 2];
            float Dr = right_plane_data[u*4 + 3];

            // Range Uncertainty
            cv::Vec4f intersect_point_bot;
            Util::intersect(Al, Bl, Cl, Dl, cam_data.nmap_nn.at<cv::Vec3f>(v,u), intersect_point_bot);
            cv::Vec4f intersect_point_top;
            Util::intersect(Ar, Br, Cr, Dr, cam_data.nmap_nn.at<cv::Vec3f>(v,u), intersect_point_top);
            float range_unc = sqrt(pow(intersect_point_bot[0]-intersect_point_top[0], 2)
                                   + pow(intersect_point_bot[1]-intersect_point_top[1], 2)
                                   + pow(intersect_point_bot[2]-intersect_point_top[2], 2) );
            surface_unc.at<float>(v,u) = range_unc;

            /*
            // Rays
            cv::Vec3f left_ray = cam_data.nmap_nn_xoffset.at<cv::Vec3f>(v,u);
            cv::Vec3f right_ray = cam_data.nmap_nn_xoffset.at<cv::Vec3f>(v,u+1);

            // Z Uncertaintiy
            cv::Vec4f zintersect_point_bot;
            Util::intersect(Al, Bl, Cl, Dl, right_ray, zintersect_point_bot);
            cv::Vec4f zintersect_point_top;
            Util::intersect(Ar, Br, Cr, Dr, left_ray, zintersect_point_top);
            float range_unc = sqrt(pow(zintersect_point_bot[0]-zintersect_point_top[0], 2)
                                   + pow(zintersect_point_bot[1]-zintersect_point_top[1], 2)
                                   + pow(zintersect_point_bot[2]-zintersect_point_top[2], 2) );
            */

            /*
            if(v == surface_pts.size().height/2) {
                if(std::isnan(laser_angles[u])) continue;
                std::cout << Al << " " << Bl << " " << Cl << " " << Dl << std::endl;
                std::cout << Ar << " " << Br << " " << Cr << " " << Dr << std::endl;
                std::cout << cam_data.nmap_nn.at<cv::Vec3f>(v,u) << std::endl;
                std::cout << intersect_point_bot << std::endl;
                std::cout << intersect_point_top << std::endl;
                std::cout << range_unc << " " << laser_angles[u] << std::endl;
                exit(-1);
            }
            */

        }

    }

    auto end = std::chrono::steady_clock::now();
    //std::cout << "Elapsed time in milliseconds : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    return surface_data;
}

void DatumProcessor::computeDepthHits(std::pair<cv::Mat,cv::Mat>& surface_data, const cv::Mat& depth_img, const Datum& cam_data){
    cv::Mat& surface_pts = surface_data.first;
    cv::Mat& surface_unc = surface_data.second;
    if(surface_pts.size() != depth_img.size()) throw std::runtime_error("Error");

    // Test against depth map
    for(int v=0; v<surface_pts.size().height; v++){
        for(int u=0; u<surface_pts.size().width; u++){
            cv::Vec4f& coord3D = surface_pts.at<cv::Vec4f>(v,u);
            float unc = surface_unc.at<float>(v,u);
            if(std::isnan(unc)) continue;
            float zval = depth_img.at<float>(v,u);
            float surface_range = sqrt(coord3D(0)*coord3D(0) + coord3D(1)*coord3D(1) + coord3D(2)*coord3D(2));
            float depth_range = zval/cam_data.ztoramap.at<float>(v,u);
            //float error = fabs(coord3D[2] - zval);
            float error = fabs(depth_range - surface_range);
            float color = 0;
            if(error >= 0) color = 255. - 255.*(error/unc);
            //(error/unc)
            if(error < unc) coord3D[3] = int(color); // Check this with joe
        }
    }
}

void DatumProcessor::processPointsT(const Eigen::MatrixXf& input_pts, const cv::Mat& depth_img, std::string cam_name, std::string laser_name, cv::Mat& image, pcl::PointCloud<pcl::PointXYZRGB>& cloud, bool compute_cloud){
    bool debug = false;
    auto begin = std::chrono::steady_clock::now();
    auto beginf = std::chrono::steady_clock::now();

    if(!set) throw std::runtime_error("Sensors not set");

    if(!cam_mapping_.count(cam_name)) throw std::runtime_error("No such camera name");
    Datum& cam_data = *(c_datums_[cam_mapping_[cam_name]].get());

    if(!cam_data.laser_data.count(laser_name)) throw std::runtime_error("No such laser name");
    Laser& laser_data = cam_data.laser_data[laser_name];

    // Convert to vec
    std::vector<Point2D> pts(input_pts.rows());
    for(int i=0; i<pts.size(); i++){
        pts[i] = Point2D(input_pts(i, 0),input_pts(i, 2));
    }

    // Check points inside sensors
    auto good_inds = checkPoints(pts, cam_data, laser_data);
    //std::cout << "A = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[µs]" << std::endl;
    //begin = std::chrono::steady_clock::now();

    // Get true points based on cam ray intersections (Slow)
    auto design_pts = findCameraIntersectionsOpt(cam_data, good_inds, pts);
    //std::cout << "B = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[µs]" << std::endl;
    //begin = std::chrono::steady_clock::now();

    // Surface Pts
    auto surface_data = calculateSurface(design_pts, cam_data, laser_data);
    auto& surface_pts = surface_data.first;
    //std::cout << "C = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[µs]" << std::endl;
    //begin = std::chrono::steady_clock::now();

    // Compute hit
    computeDepthHits(surface_data, depth_img, cam_data);
    //std::cout << "D = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[µs]" << std::endl;
    //begin = std::chrono::steady_clock::now();

    // Assign
    surface_pts.copyTo(image);

    // Cloud Compute
    if(!compute_cloud) return;

    // Downsample
    float downsample = 2.;
    if(downsample > 1){
        float voxelize = 1./((float)downsample);
        cv::resize(surface_pts, surface_pts, cv::Size(), voxelize, voxelize, cv::INTER_NEAREST);
    }
    //std::cout << "F = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[µs]" << std::endl;
    //begin = std::chrono::steady_clock::now();

//    // Store it
//    for(int v=0; v<surface_pts.size().height; v++){
//        for(int u=0; u<surface_pts.size().width; u++){
//            int index = v*surface_pts.size().width + u;
//            const cv::Vec4f& coord3D = surface_pts.at<cv::Vec4f>(v,u);
//            if(std::isnan(coord3D[0])) continue;
//            pcl::PointXYZRGB point;
//            point.x = coord3D[0];
//            point.y = coord3D[1];
//            point.z = coord3D[2];
//            point.r = 1.0;
//            point.g = coord3D[3];
//            point.b = 1.0;
//            cloud.push_back(point);
//        }
//    }

    cloud.resize(surface_pts.size().width*surface_pts.size().height);
    for(int v=0; v<surface_pts.size().height; v++){
        for(int u=0; u<surface_pts.size().width; u++){
            int index = v*surface_pts.size().width + u;
            const cv::Vec4f& coord3D = surface_pts.at<cv::Vec4f>(v,u);
            cloud[index].x = coord3D[0];
            cloud[index].y = coord3D[1];
            cloud[index].z = coord3D[2];
            cloud[index].r = 1.0;
            cloud[index].g = coord3D[3];
            cloud[index].b = 1.0;
        }
    }

}

void DatumProcessor::processTest(Eigen::MatrixXf& input_pts, std::string cam_name, std::string laser_name, std::shared_ptr<Output>& output, int mode){
    if(!cam_mapping_.count(cam_name)) throw std::runtime_error("No such camera name");
    Datum& cam_data = *(c_datums_[cam_mapping_[cam_name]].get());

    if(!cam_data.laser_data.count(laser_name)) throw std::runtime_error("No such laser name");
    Laser& laser_data = cam_data.laser_data[laser_name];

    // Convert to vec
    std::vector<Point2D> pts(input_pts.rows());
    if(mode == 0){
        for(int i=0; i<pts.size(); i++){
            pts[i] = Point2D(input_pts(i, 0),input_pts(i, 2));
        }
    }else if(mode == 1){
        for(int i=0; i<pts.size(); i++){
            pts[i] = Point2D(input_pts(i, 0),input_pts(i, 1));
        }
    }

    // Check points inside sensors
    auto good_inds = checkPoints(pts, cam_data, laser_data);

    // Get true points based on cam ray intersections (Slow)
    //auto design_pts = findCameraIntersections(cam_data, good_inds, pts);
    Eigen::Matrix4Xf design_pts = findCameraIntersectionsOpt(cam_data, good_inds, pts);

    // Get Angles
    std::shared_ptr<Angles> angles = calculateAngles(design_pts, cam_data, laser_data, true, true);

//        // Store rays
//        Eigen::MatrixXf rays;
//        rays.resize(cam_data.imgw, 3);
//        for(int u=0; u<cam_data.imgw; u++){
//            rays(u,0) = cam_data.midrays[0].at<float>(0,u);
//            rays(u,1) = cam_data.midrays[1].at<float>(0,u);
//            rays(u,2) = cam_data.midrays[2].at<float>(0,u);
//        }

    // Store
    output->spline = input_pts;
    output->output_pts = angles->output_pts;
    output->angles = angles->angles;
    output->velocities = angles->velocities;
    output->accels = angles->accels;
    //output->laser_rays = rays;
}
