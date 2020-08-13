#ifndef PLANNING_H
#define PLANNING_H

#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <common.h>
#include <dprocessor.h>

#define PI 3.14159265
#define MAX_RAYS 1500
#define MAX_NODES_PER_RAY 500

namespace lc {

#define INF std::numeric_limits<float>::infinity()

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

template <bool MAX>
class Trajectory {
public:
    float cost;  // sum of costs to be maximized or minimized
    float las;  // sum of squares of laser angle deviation to be minimized

    Node* pNode;  // node the trajectory starts from
    Trajectory* pSubTraj;  // the rest of the sub-trajectory;

    Trajectory();
    Trajectory(Node* pNode_, const Eigen::MatrixXf& cmap);
    Trajectory(Node* pNode_, Trajectory* pSubTraj_, const Eigen::MatrixXf& cmap);
    ~Trajectory();

    bool operator< (const Trajectory& o);
    bool operator> (const Trajectory& o);

};

class Interpolator {
public:
    virtual std::pair<int, int> getCmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const = 0;
    virtual bool isCmapShapeValid(int nrows, int ncols) const = 0;
};

class CartesianNNInterpolator : public Interpolator {
private:
    int cmap_w_, cmap_h_;
    float x_min_, x_max_, z_min_, z_max_;
public:
    CartesianNNInterpolator(int cmap_w, int cmap_h, float x_min, float x_max, float z_min, float z_max);
    std::pair<int, int> getCmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const override;
    bool isCmapShapeValid(int nrows, int ncols) const override;
};

class PolarIdentityInterpolator : public Interpolator {
private:
    int num_camera_rays_, num_ranges_;
public:
    PolarIdentityInterpolator(int num_camera_rays, int num_ranges);
    std::pair<int, int> getCmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const override;
    bool isCmapShapeValid(int nrows, int ncols) const override;
};

template <bool MAX>
class Planner {
private:
    bool debug_;
    std::shared_ptr<DatumProcessor> datumProcessor_;
    std::vector<float> camera_angles_;
    float max_d_las_angle_;
    Eigen::Matrix4f cam_to_laser_;

    Node graph_[MAX_RAYS][MAX_NODES_PER_RAY];
    Trajectory<MAX> dp_[MAX_RAYS][MAX_NODES_PER_RAY];
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

    std::vector<std::pair<float, float>> optimizedDesignPts(Eigen::MatrixXf cmap);

    std::vector<std::vector<std::pair<Node, int>>> getVectorizedGraph();
};

// explicit instantiations
template class Planner<true>;
template class Planner<false>;

}

#endif
