#include "planning.h"
#include <limits>

using namespace lc;

// ----------- UTILS -----------
float deg2rad(float deg) {
    return deg * PI / 180;
}
float rad2deg(float rad) {
    return rad * 180 / PI;
}
// -----------------------------

Planner::Planner(std::shared_ptr<DatumProcessor> datumProcessor,
                 const std::vector<float>& ranges,
                 std::shared_ptr<Interpolator> interpolator,
                 bool debug) : datumProcessor_(datumProcessor), ranges_(ranges), interpolator_(interpolator), debug_(debug){
    std::shared_ptr<Datum> c_datum = datumProcessor_->getCDatum("camera01");  // assume only one camera named camera01
    Laser laser = c_datum->laser_data["laser01"];  // assume only one camera named camera01

    camera_angles_ = datumProcessor_->getCDatum("camera01")->valid_angles;
    num_camera_rays_ = camera_angles_.size();
    num_nodes_per_ray_ = ranges_.size();
    max_d_las_angle_ = laser.laser_limit * laser.laser_timestep;
    laser_to_cam_ = laser.laser_to_cam;

    if (debug_) {
        std::cout << std::setprecision(4)
                  << "PYLC_PLANNER: Max change in laser angle: " << max_d_las_angle_ << "°" << std::endl;
    }

    constructGraph();
}

void Planner::constructGraph() {
    // Add nodes in the graph.
    for (int ray_i = 0; ray_i < num_camera_rays_; ray_i++)
        for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++) {
            float r = ranges_[range_i];
            float theta_cam = camera_angles_[ray_i];
            float x = r * std::sin(deg2rad(theta_cam));
            float z = r * std::cos(deg2rad(theta_cam));

            // Compute laser angle.
            Eigen::Vector4f xyz1_cam(x, 0.0f, z, 1.0f);
            Eigen::Vector4f xyz1_las = laser_to_cam_ * xyz1_cam;
            float x_las = xyz1_las(0), z_las = xyz1_las(2);
            float theta_las = rad2deg(std::atan2(x_las, z_las));

            std::pair<int, int> k = interpolator_->getUmapIndex(x, z, r, theta_cam, theta_las, ray_i, range_i);
            int ki = k.first, kj = k.second;

            graph_[ray_i][range_i].fill(x, z, r, theta_cam, theta_las, ki, kj);
        }

    // Add edges in the graph.
    for (int ray_i = 0; ray_i < num_camera_rays_ - 1; ray_i++) {
        Node* ray_prev = graph_[ray_i];
        Node* ray_next = graph_[ray_i + 1];

        for (int prev_i = 0; prev_i < num_nodes_per_ray_; prev_i++) {
            Node &node_prev = ray_prev[prev_i];
            for (int next_i = 0; next_i < num_nodes_per_ray_; next_i++) {
                Node &node_next = ray_next[next_i];

                float d_theta_las = node_next.theta_las - node_prev.theta_las;
                bool is_neighbor = (-max_d_las_angle_ <= d_theta_las) && (d_theta_las <= max_d_las_angle_);
                if (is_neighbor)
                    node_prev.edges.emplace_back(ray_i + 1, next_i);
            }
        }
    }
}

std::vector<std::pair<float, float>> Planner::optimizedDesignPts(Eigen::MatrixXf umap) {
    // Check if umap shape is as expected.
    if (!interpolator_->isUmapShapeValid(umap.rows(), umap.cols()))
        throw std::invalid_argument(std::string("PYLC_PLANNER: Unexpected umap shape (")
                                    + std::to_string(umap.size())
                                    + std::string(")."));

    // Backward pass.
    for (int ray_i = num_camera_rays_ - 1; ray_i >= 0; ray_i--) {
        for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++) {
            Node* pNode = &(graph_[ray_i][range_i]);

            // Trajectory starting from and ending at here.
            dp_[ray_i][range_i] = Trajectory(pNode, umap);

            // Select best sub-trajectory from valid neighbors.
            for (std::pair<int, int> edge : pNode->edges) {
                Trajectory* pSubTraj = &(dp_[edge.first][edge.second]);
                Trajectory traj(pNode, pSubTraj, umap);
                if (traj > dp_[ray_i][range_i])
                    dp_[ray_i][range_i] = traj;
            }
        }
    }

    // Select overall best trajectory.
    Trajectory best_traj;
    for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++)
        if (dp_[0][range_i] > best_traj)
            best_traj = dp_[0][range_i];

    if (debug_) {
        std::cout << std::fixed << std::setprecision(3)
                  << "PYLC_PLANNER: Optimal uncertainty  : " << best_traj.unc << std::endl
                  << "              Optimal laser penalty: " << best_traj.las << std::endl
                  ;
    }

    // Forward pass.
    std::vector<std::pair<float, float>> design_pts;
    while (true) {
        // Current design point.
        design_pts.emplace_back(best_traj.pNode->x, best_traj.pNode->z);

        if (!best_traj.pSubTraj)  // trajectory ends here
            break;

        best_traj = *(best_traj.pSubTraj);
    }

    return design_pts;
}

std::vector<std::vector<std::pair<Node, int>>> Planner::getVectorizedGraph() {
    std::vector<std::vector<std::pair<Node, int>>> m;

    // Copy 2D array to matrix.
    for (int ray_i = 0; ray_i < num_camera_rays_; ray_i++) {
        m.emplace_back();
        for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++) {
            Node& node = graph_[ray_i][range_i];
            m[ray_i].emplace_back(node, node.edges.size());
        }
    }
    return m;
}

Planner::~Planner() = default;

Node::Node() = default;

Node::~Node() {
    edges.clear();
}

void Node::fill(float x_, float z_, float r_, float theta_cam_, float theta_las_, long ki_, long kj_) {
    x = x_;
    z = z_;
    r = r_;
    theta_cam = theta_cam_;
    theta_las = theta_las_;
    ki = ki_;
    kj = kj_;

    edges = std::vector<std::pair<int, int>>();
}

Trajectory::Trajectory() {
    pNode = nullptr;
    pSubTraj = nullptr;
    unc = 0.0f;
    las = 0.0f;
}

Trajectory::Trajectory(Node* pNode_, const Eigen::MatrixXf& umap) {
    // Start node.
    pNode = pNode_;

    // Sub-trajectory.
    pSubTraj = nullptr;

    // Uncertainty.
    if ((pNode->ki != -1) && (pNode->kj != -1))
        unc = umap(pNode->ki, pNode->kj);
    else
        unc = 0.0f;

    // Laser penalty.
    las = 0.0f;
}

Trajectory::Trajectory(Node* pNode_, Trajectory* pSubTraj_, const Eigen::MatrixXf& umap) : Trajectory(pNode_, umap) {
    // Start Node : delegated.

    // Sub-trajectory.
    pSubTraj = pSubTraj_;

    // Uncertainty.
    // Initialized from delegation.
    unc += pSubTraj->unc;

    // Laser angle penalty : sum of squares of laser angle changes.
    float d_theta_cam = pSubTraj->pNode->theta_las - pNode->theta_las;
    las = d_theta_cam * d_theta_cam + pSubTraj->las;
}

bool Trajectory::operator<(const Trajectory& t) {
    if (unc < t.unc)
        return true;
    else if (unc == t.unc)
        return las > t.las;
    else
        return false;
}

bool Trajectory::operator>(const Trajectory& t) {
    if (unc > t.unc)
        return true;
    else if (unc == t.unc)
        return las < t.las;
    else
        return false;
}

Trajectory::~Trajectory() = default;

CartesianNNInterpolator::CartesianNNInterpolator(int umap_w, int umap_h,
                                                 float x_min, float x_max, float z_min, float z_max) {
    umap_w_ = umap_w;
    umap_h_ = umap_h;
    x_min_ = x_min;
    x_max_ = x_max;
    z_min_ = z_min;
    z_max_ = z_max;
}

std::pair<int, int> CartesianNNInterpolator::getUmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const {
    // This function assumes that uncertainty_map is a 2D grid with evenly spaced xs and zs.
    // It also assumes that X and Z are increasing with a fixed increment.

    float x_incr = (x_max_ - x_min_) / float(umap_h_ - 1);
    float z_incr = (z_max_ - z_min_) / float(umap_w_ - 1);

    // Convert to pixel coordinates.
    long ki = std::lround((x - x_min_) / x_incr);
    long kj = std::lround((z - z_min_) / z_incr);

    if (ki < 0 || ki >= umap_h_)
        ki = -1;  // means that this is outside the umap grid

    if (kj < 0 || kj >= umap_w_)
        kj = -1;  // means that this is outside the umap grid

    return {ki, kj};
}

bool CartesianNNInterpolator::isUmapShapeValid(int nrows, int ncols) const {
    return (nrows == umap_h_) && (ncols == umap_w_);
}

PolarIdentityInterpolator::PolarIdentityInterpolator(int num_camera_rays, int num_ranges)
    : num_camera_rays_(num_camera_rays), num_ranges_(num_ranges) {}

std::pair<int, int> PolarIdentityInterpolator::getUmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const {
    return {range_i, ray_i};
}

bool PolarIdentityInterpolator::isUmapShapeValid(int nrows, int ncols) const {
    return (nrows == num_ranges_) && (ncols == num_camera_rays_);
}
