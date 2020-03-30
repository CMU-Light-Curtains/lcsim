#ifndef ALGO_H
#define ALGO_H

#include <string>
#include <iostream>
#include <common.h>
#include <poly34.h>

namespace lc{

class Algo{
public:
    static Point2D nn(Point2D pt, const std::vector<Point2D>& test_pts);
    static void angleSort(std::vector<Point2D>& pts);
    static std::vector<std::vector<Point2D>> listSort(const std::vector<std::vector<Point2D>>& list);
    static bool closestDistance(Eigen::MatrixXf& design_pts, Eigen::MatrixXf& targets, float threshold=0.1, bool debug=false);
    static void removeNan(std::vector<float>& x);
    static std::vector<std::vector<int>> splitVector(const std::vector<int>& idx, int split_count);
    template< class T > static void reorder(std::vector<T>& vA, const std::vector<size_t>& vOrder){
        assert(vA.size() == vOrder.size());
        std::vector<T> vCopy = vA; // Can we avoid this?
        for(int i = 0; i < vOrder.size(); ++i)
            vA[i] = vCopy[ vOrder[i] ];
    }

    static int convolve_sse(float* in, int input_length, float* kernel,	int kernel_length, float* out);
    static int convolve_naive(float* in, int input_length, float* kernel,	int kernel_length, float* out);
    static std::vector<float> convolve(std::vector<float>& input, std::vector<float>& kernel, int mode = 0);
    static float squaredSum(std::vector<float>& x);
    static std::vector<float> arange(float start, float stop, float step = 1, bool include=false);
    static std::vector<float> linspace(float a, float b, size_t N);
    static Eigen::VectorXf gaussian(const Eigen::VectorXf& x, float mu, float sig, float power=2.);

    static void setCol(Eigen::MatrixXf& matrix, int colnum, float val);
    static void eigenXSort(Eigen::MatrixXf& matrix);
    static std::vector<int> eigenAngleSort(Eigen::MatrixXf& matrix);
    static void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove);
    static Eigen::MatrixXf customSort(const Eigen::MatrixXf& matrix, const std::vector<int>& custom_order);
    static void removeRows(Eigen::MatrixXf& matrix, std::vector<int>& to_remove);

    static std::vector<std::vector<std::vector<int>>> generateContPerms(int index_count, int split_count);
    static std::vector<std::vector<std::vector<int>>> getPartitions(const std::vector<int>& elements);
    static std::vector<std::vector<int>> getPermutations(const std::vector<int>& elements);

    static Eigen::MatrixXf fitBSpline(const Eigen::MatrixXf& input_pts_x, std::shared_ptr<SplineParamsVec>& splineParamsVec, bool computeCurve=true);
    static Eigen::MatrixXf solveT(const std::shared_ptr<SplineParamsVec>& splineParamsVec, Eigen::MatrixXf inputPts);

};

}

#endif