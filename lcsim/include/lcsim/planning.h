#ifndef PLANNING_H
#define PLANNING_H

#include <string>
#include <iostream>
#include <limits>
#include <cmath>
#include <common.h>
#include <dprocessor.h>

#define PI 3.14159265
#define MAX_RAYS 1500
#define MAX_NODES_PER_RAY 500

namespace lc {


struct Node {
public:
    float x, z;
    float r, theta_cam, theta_las;
    long ki, kj;

    std::vector<std::pair<int, int>> edges;

    Node();
    ~Node();

    void fill(float x_, float z_, float r_, float theta_cam_, float theta_las_, long ki_, long kj_);
};

class Trajectory {
public:
    float unc;  // sum of uncertainties
    float las;  // sum of squares of laser angle deviation

    Node* pNode;  // node the trajectory starts from
    Trajectory* pSubTraj;  // the rest of the sub-trajectory;

    Trajectory();
    Trajectory(Node* pNode_, const Eigen::MatrixXf& umap);
    Trajectory(Node* pNode_, Trajectory* pSubTraj_, const Eigen::MatrixXf& umap);
    ~Trajectory();

    bool operator< (const Trajectory& o);
    bool operator> (const Trajectory& o);

};

class Interpolator {
public:
    virtual std::pair<int, int> getUmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const = 0;
    virtual bool isUmapShapeValid(int nrows, int ncols) const = 0;
};

class CartesianNNInterpolator : public Interpolator {
private:
    int umap_w_, umap_h_;
    float x_min_, x_max_, z_min_, z_max_;
public:
    CartesianNNInterpolator(int umap_w, int umap_h, float x_min, float x_max, float z_min, float z_max);
    std::pair<int, int> getUmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const override;
    bool isUmapShapeValid(int nrows, int ncols) const override;
};

class PolarIdentityInterpolator : public Interpolator {
private:
    int num_camera_rays_, num_ranges_;
public:
    PolarIdentityInterpolator(int num_camera_rays, int num_ranges);
    std::pair<int, int> getUmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const override;
    bool isUmapShapeValid(int nrows, int ncols) const override;
};

class Planner {
private:
    bool debug_;
    std::shared_ptr<DatumProcessor> datumProcessor_;
    std::vector<float> camera_angles_;
    float max_d_las_angle_;
    Eigen::Matrix4f laser_to_cam_;

    Node graph_[MAX_RAYS][MAX_NODES_PER_RAY];
    Trajectory dp_[MAX_RAYS][MAX_NODES_PER_RAY];
    const std::vector<float> ranges_;
    std::shared_ptr<Interpolator> interpolator_;
    int num_camera_rays_, num_nodes_per_ray_;

    void constructGraph();

public:
    Planner(std::shared_ptr<DatumProcessor> datumProcessor,
            const std::vector<float>& ranges,
            std::shared_ptr<Interpolator> interpolator,
            bool debug);
    ~Planner();

    std::vector<std::pair<float, float>> optimizedDesignPts(Eigen::MatrixXf umap);

    std::vector<std::vector<std::pair<Node, int>>> getVectorizedGraph();
};

}

#endif
