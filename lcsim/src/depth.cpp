#include "depth.h"

using namespace lc;

Eigen::MatrixXf Depth::upsampleLidar(const Eigen::MatrixXf &lidardata_cam, std::map<std::string, float> &params) {
    int total_vbeams = 128;
    int total_hbeams = 1500;
    float vbeam_fov = 0.2;
    float hbeam_fov = 0.08;
    float phioffset = 10;

    float scale = params["upsample"];
    if(params.count("total_vbeams")) total_vbeams = (int)(params["total_vbeams"]);
    if(params.count("total_hbeams")) total_hbeams = (int)(params["total_hbeams"]);
    if(params.count("vbeam_fov")) vbeam_fov = params["vbeam_fov"];
    if(params.count("hbeam_fov")) hbeam_fov = params["hbeam_fov"];

    float vscale = 1.;
    float hscale = 1.;
    int vbeams = (int)(total_vbeams*vscale);
    int hbeams = (int)(total_hbeams*hscale);
    float vf = vbeam_fov/vscale;
    float hf = hbeam_fov/hscale;
    cv::Mat rmap = cv::Mat(vbeams, hbeams, CV_32FC1, cv::Scalar(0.));

    // Cast to Angles
    Eigen::MatrixXf rtp = Eigen::MatrixXf::Zero(lidardata_cam.rows(), 3);
    rtp.col(0) = Eigen::sqrt(Eigen::pow(lidardata_cam.col(0).array(),2) + Eigen::pow(lidardata_cam.col(1).array(),2) + Eigen::pow(lidardata_cam.col(2).array(),2));
    rtp.col(1) = (Eigen::atan(lidardata_cam.col(0).cwiseQuotient(lidardata_cam.col(2)).array()) * (180./M_PI));
    rtp.col(2) = (Eigen::asin(lidardata_cam.col(1).cwiseQuotient(rtp.col(0)).array()) * (180./M_PI)) - phioffset;

    // Bin Data
    for(int i=0; i<rtp.rows(); i++){
        float r = rtp(i,0);
        float theta = rtp(i,1);
        float phi = rtp(i,2);
        int thetabin = (int)(((theta/hf) + hbeams/2)+0.5);
        int phibin = (int)(((phi/vf) + vbeams/2)+0.5);
        if(thetabin < 0 || thetabin >= hbeams || phibin < 0 || phibin >= vbeams) continue;
        float current_r = rmap.at<float>(phibin, thetabin);
        if((r < current_r) || (current_r == 0))
            rmap.at<float>(phibin, thetabin) = r;
    }

    // Upsample
    vscale = vscale*scale;
    hscale = hscale*scale;
    vbeams = (int)(total_vbeams*vscale);
    hbeams = (int)(total_hbeams*hscale);
    vf = vbeam_fov/vscale;
    hf = hbeam_fov/hscale;
    cv::resize(rmap, rmap, cv::Size(0,0), hscale, vscale, cv::INTER_NEAREST);

    // Regenerate
    Eigen::MatrixXf xyz_new = Eigen::MatrixXf::Ones(rmap.size().width * rmap.size().height, 4);
    for(int phibin=0; phibin<rmap.size().height; phibin++){
        for(int thetabin=0; thetabin<rmap.size().width; thetabin++){
            int i = phibin*rmap.size().width + thetabin;
            float phi = ((phibin - (vbeams/2.))*vf + phioffset)*(M_PI/180.);
            float theta = ((thetabin - (hbeams / 2.))*hf)*(M_PI/180.);
            float r = rmap.at<float>(phibin, thetabin);
            xyz_new(i,0) = r*cos(phi)*sin(theta);
            xyz_new(i,1) = r*sin(phi);
            xyz_new(i,2) = r*cos(phi)*cos(theta);
        }
    }

    return xyz_new;
}

Eigen::MatrixXf Depth::generateDepth(const Eigen::MatrixXf &lidardata, const Eigen::MatrixXf &intr_raw, const Eigen::MatrixXf &M_lidar2cam, int width, int height, std::map<std::string, float> &params) {
    float upsample = params["upsample"];
    int filtering = (int)(params["filtering"]);

    // Transform to Camera Frame
    Eigen::MatrixXf lidardata_cam = (M_lidar2cam * lidardata.transpose()).transpose();

    // Remove points behind camera
    Eigen::MatrixXf lidardata_cam_cleaned = Eigen::MatrixXf::Zero(lidardata_cam.rows(), lidardata_cam.cols());
    int j=0;
    for(int i=0; i<lidardata_cam.rows(); i++){
        auto z = lidardata_cam(i,2);
        if(z >= 0.1){
            lidardata_cam_cleaned.row(j) = lidardata_cam.row(i);
            j++;
        }
    }
    lidardata_cam = lidardata_cam_cleaned.block(0,0,j,lidardata_cam.cols());

    // Upsample
    if(upsample){
        lidardata_cam = upsampleLidar(lidardata_cam, params);
    }

    // Project and Generate Pixels
    Eigen::MatrixXf lidardata_cam_proj = (intr_raw * lidardata_cam.transpose()).transpose();
    lidardata_cam_proj.col(0) = lidardata_cam_proj.col(0).cwiseQuotient(lidardata_cam_proj.col(2));
    lidardata_cam_proj.col(1) = lidardata_cam_proj.col(1).cwiseQuotient(lidardata_cam_proj.col(2));
    lidardata_cam_proj.col(2) = lidardata_cam.col(2);

    // Z Buffer assignment
    Eigen::MatrixXf dmap_raw = Eigen::MatrixXf::Zero(height, width);
    for(int i=0; i<lidardata_cam_proj.rows(); i++){
        int u = (int)(lidardata_cam_proj(i,0) - 0.5);
        int v = (int)(lidardata_cam_proj(i,1) - 0.5);
        if(u < 0 || u >= width || v < 0 || v >= height) continue;
        float z = lidardata_cam_proj(i,2);
        float current_z = dmap_raw(v,u);
        if((z < current_z) || (current_z == 0))
            dmap_raw(v,u) = z;
    }

    // Filtering
    Eigen::MatrixXf dmap_cleaned = Eigen::MatrixXf::Zero(height, width);
    int offset = filtering;
    for(int v=offset; v<height-offset-1; v++){
        for(int u=offset; u<width-offset-1; u++){
            float z = dmap_raw(v,u);
            bool bad = false;

            // Check neighbours
            for(int vv=v-offset; vv<v+offset+1; vv++){
                for(int uu=u-offset; uu<u+offset+1; uu++){
                    if(vv == v && uu == u) continue;
                    float zn = dmap_raw(vv,uu);
                    if(zn == 0) continue;
                    if((zn-z) < -1){
                        bad = true;
                        break;
                    }
                }
            }

            if(!bad){
                dmap_cleaned(v,u) = z;
            }
        }
    }

    return dmap_cleaned;
}

std::vector<cv::Mat> Depth::transformPoints(const Eigen::MatrixXf &lidardata, const Eigen::VectorXf &thickdata, const Eigen::MatrixXf &intr_raw, const Eigen::MatrixXf &M_lidar2cam, int width, int height, std::map<std::string, float>& params) {
    int filtering = (int)(params["filtering"]);

    // Extract Intensity and Points
    Eigen::MatrixXf lidardata_points = lidardata;
    lidardata_points.col(3) = Eigen::VectorXf::Ones(lidardata.rows());
    Eigen::VectorXf lidardata_intensity = lidardata.col(3);

    // Transform to Camera Frame
    Eigen::MatrixXf lidardata_cam = (M_lidar2cam * lidardata_points.transpose()).transpose();

    // Project and Generate Pixels
    Eigen::MatrixXf lidardata_cam_proj = (intr_raw * lidardata_cam.transpose()).transpose();
    lidardata_cam_proj.col(0) = lidardata_cam_proj.col(0).cwiseQuotient(lidardata_cam_proj.col(2));
    lidardata_cam_proj.col(1) = lidardata_cam_proj.col(1).cwiseQuotient(lidardata_cam_proj.col(2));
    lidardata_cam_proj.col(2) = lidardata_cam.col(2);

    // Z Buffer assignment
    Eigen::MatrixXf dmap_raw = Eigen::MatrixXf::Zero(height, width);
    Eigen::MatrixXf imap_raw = Eigen::MatrixXf::Zero(height, width);
    Eigen::MatrixXf tmap_raw = Eigen::MatrixXf::Zero(height, width);
    for(int i=0; i<lidardata_cam_proj.rows(); i++){
        int u = (int)(lidardata_cam_proj(i,0) - 0.5);
        int v = (int)(lidardata_cam_proj(i,1) - 0.5);
        if(u < 0 || u >= width || v < 0 || v >= height) continue;
        float z = lidardata_cam_proj(i,2);
        if(std::isnan(z)) continue;
        float intensity = lidardata_intensity(i);
        float thickness = thickdata(i);
        float current_z = dmap_raw(v,u);
        //if(intensity == 0 || std::isnan(intensity)) continue;
        if((z < current_z) || (current_z == 0)) {
            dmap_raw(v, u) = z;
            imap_raw(v, u) = intensity;
            tmap_raw(v,u) = thickness;
        }
    }

    // Filtering
    Eigen::MatrixXf dmap_mask = Eigen::MatrixXf::Ones(height, width);
    int offset = filtering;
    if(offset) {
        for (int v = offset; v < height - offset - 1; v++) {
            for (int u = offset; u < width - offset - 1; u++) {
                float z = dmap_raw(v, u);
                bool bad = false;

                // Check neighbours
                for (int vv = v - offset; vv < v + offset + 1; vv++) {
                    for (int uu = u - offset; uu < u + offset + 1; uu++) {
                        if (vv == v && uu == u) continue;
                        float zn = dmap_raw(vv, uu);
                        if (zn == 0) continue;
                        if ((zn - z) < -1) {
                            bad = true;
                            break;
                        }
                    }
                }

                if (bad) {
                    dmap_mask(v, u) = 0;
                }
            }
        }
    }


    // Apply mask
    dmap_raw = dmap_raw.cwiseProduct(dmap_mask);
    imap_raw = imap_raw.cwiseProduct(dmap_mask);
    tmap_raw = tmap_raw.cwiseProduct(dmap_mask);
    cv::Mat dmap_raw_cv, imap_raw_cv, tmap_raw_cv;
    cv::eigen2cv(dmap_raw, dmap_raw_cv);
    cv::eigen2cv(imap_raw, imap_raw_cv);
    cv::eigen2cv(tmap_raw, tmap_raw_cv);

    std::vector<cv::Mat> outputs = {dmap_raw_cv, imap_raw_cv, tmap_raw_cv};
    return outputs;
}