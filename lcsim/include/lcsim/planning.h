#ifndef PLANNING_H
#define PLANNING_H

#include <string>
#include <iostream>
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

class Planner {
public:
    bool debug_;
    std::shared_ptr<DatumProcessor> datumProcessor_;
    std::vector<float> camera_angles_;
    float max_d_las_angle_;
    Eigen::Matrix4f laser_to_cam_;

    Node graph_[MAX_RAYS][MAX_NODES_PER_RAY];
    Trajectory dp_[MAX_RAYS][MAX_NODES_PER_RAY];
    int umap_w_, umap_h_;
    float x_min_, x_max_, z_min_, z_max_;
    int camera_rays_;
    int nodes_per_ray_;

    Planner(std::shared_ptr<DatumProcessor> datumProcessor, bool debug);
    ~Planner();

    void constructGraph(int umap_w_, int umap_h_,
                        float x_min_, float x_max_, float z_min_, float z_max_,
                        int nodes_per_ray_);

    std::vector<std::pair<float, float>> optimizedDesignPts(Eigen::MatrixXf umap);

    std::vector<std::vector<std::pair<Node, int>>> getVectorizedGraph();

    std::pair<int, int> nearestNeighborIndex(float x, float z);
};

}

#endif