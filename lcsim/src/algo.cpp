#include <numeric>
#include "algo.h"

using namespace lc;

bool sortX(const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    return a(0) < b(0);
}

bool sortY(const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    return a(1) < b(1);
}

Point2D Algo::nn(Point2D pt, const std::vector<Point2D> &test_pts) {
    Point2D closest_pt;
    float closest_dist = std::numeric_limits<float>::infinity();
    for(const auto& p: test_pts){
        float dist = sqrt(pow(pt(0)-p(0),2) + pow(pt(1)-p(1),2));
        if(dist < closest_dist){
            closest_pt = p;
            closest_dist = dist;
        }
    }
    return closest_pt;
}

void Algo::angleSort(std::vector<Point2D> &pts) {
    // Sort and extract
    struct csort
    {
        inline bool operator() (const Point2D& p1, const Point2D& p2)
        {
            float angle_p0p1 = atan2(p1(1),p1(0));
            float angle_p0p2 = atan2(p2(1),p2(0));
            return angle_p0p1 > angle_p0p2;
        }
    };
    std::sort(pts.begin(), pts.end(), csort());
}

std::vector<std::vector<int>> getPermutations(const std::vector<int>& elements){
    std::vector<std::vector<int>> p;
    int n = elements.size();
    std::vector<int> elementsCpy = std::vector<int>(elements);
    int* myints = &elementsCpy[0];
    do {
        p.emplace_back(elementsCpy);
    } while ( std::next_permutation(myints,myints+n) );
    return p;
}

std::vector<std::vector<Point2D> > Algo::listSort(const std::vector<std::vector<Point2D> > &list) {
    std::vector<std::vector<Point2D>> flist;

    // Special Case
    if(list.size() <= 1){
        flist = std::vector<std::vector<Point2D>>(list);
        return flist;
    }

    // Get Permutations
    std::vector<int> indexes;
    for(int i=0; i<list.size(); i++){
        indexes.emplace_back(i);
    }
    std::vector<std::vector<int>> permutations = getPermutations(indexes);

    // Create index pair for permutations
    std::vector<std::pair<int, float>> scores(permutations.size());
    for(int i=0; i<permutations.size(); i++){
        scores[i].first = i;
        scores[i].second = std::numeric_limits<float>::infinity();
    }

    // Iterate each Permutation set
    for(int i=0; i<scores.size(); i++){

        // Iterate each element in permutation
        float cost = 0;
        for(int j=0; j<permutations[i].size()-1; j++){
            int i1 = permutations[i][j];
            int i2 = permutations[i][j+1];
            Point2D p1 = list[i1].back();
            Point2D p2 = list[i2].front();
            cost += sqrt(pow(p1(0)-p2(0),2) + pow(p1(1)-p2(1),2));
        }

        // Set cost
        scores[i].second = cost;
    }

    // Sort
    struct csort
    {
        inline bool operator() (const std::pair<int, float>& c1, const std::pair<int, float>& c2)
        {
            return (c1.second < c2.second);
        }
    };
    std::sort(scores.begin(), scores.end(), csort());

    // Create flist
    auto best_perm_index = scores.front().first;
    auto best_perm = permutations[best_perm_index];
    for(auto idx : best_perm){
        flist.emplace_back(list[idx]);
    }

    return flist;
}

bool Algo::closestDistance(Eigen::MatrixXf& design_pts, Eigen::MatrixXf& targets, float threshold, bool debug){
    for(int i=0; i<targets.rows(); i++){
        float lowest_l2 = std::numeric_limits<float>::infinity();
        //Eigen::Vector2f closest_point;
        for(int j=0; j<design_pts.rows(); j++){
            if(std::isnan(design_pts(j,0))) continue;
            float l2 = sqrt(pow(targets(i,0)-design_pts(j,0),2) + pow(targets(i,1)-design_pts(j,2),2));
            if(l2 < lowest_l2){
                lowest_l2 = l2;
                //closest_point = Eigen::Vector2f(design_pts(j,0), design_pts(j,2));
            }
            //std::cout << "[" << design_pts(j,0) << ", " << design_pts(j,2) << "]" << std::endl;
        }
        if(lowest_l2 > threshold) return true;
    }
    return false;
}

void Algo::removeNan(std::vector<float> &x) {
    x.erase(std::remove_if(std::begin(x), std::end(x), [](const float& value) { return std::isnan(value); }), std::end(x));
}

int Algo::convolve_sse(float* in, int input_length,
                 float* kernel,	int kernel_length, float* out)
{
    float* in_padded = (float*)(alloca(sizeof(float) * (input_length + 8)));

    __m128* kernel_many = (__m128*)(alloca(16 * kernel_length));
    __m128 block;

    __m128 prod;
    __m128 acc;

    // surrounding zeroes, before and after
    _mm_storeu_ps(in_padded, _mm_set1_ps(0));
    memcpy(&in_padded[4], in, sizeof(float) * input_length);
    _mm_storeu_ps(in_padded + input_length + 4, _mm_set1_ps(0));

    // Repeat each kernal value across a 4-vector
    int i;
    for (i = 0; i < kernel_length; i++) {
        kernel_many[i] = _mm_set1_ps(kernel[i]); // broadcast
    }

    for (i = 0; i < input_length + kernel_length - 4; i += 4) {

        // Zero the accumulator
        acc = _mm_setzero_ps();

        int startk = i > (input_length - 1) ? i - (input_length - 1) : 0;
        int endk = (i + 3) < kernel_length ? (i + 3) : kernel_length - 1;

        /* After this loop, we have computed 4 output samples
        * for the price of one.
        * */
        for (int k = startk; k <= endk; k++) {

            // Load 4-float data block. These needs to be an unaliged
            // load (_mm_loadu_ps) as we step one sample at a time.
            block = _mm_loadu_ps(in_padded + 4 + i - k);
            prod = _mm_mul_ps(block, kernel_many[k]);

            // Accumulate the 4 parallel values
            acc = _mm_add_ps(acc, prod);
        }
        _mm_storeu_ps(out + i, acc);
    }

    // Left-overs
    for (; i < input_length + kernel_length - 1; i++) {

        out[i] = 0.0;
        int startk = i >= input_length ? i - input_length + 1 : 0;
        int endk = i < kernel_length ? i : kernel_length - 1;
        for (int k = startk; k <= endk; k++) {
            out[i] += in[i - k] * kernel[k];
        }
    }

    return 0;
}

int Algo::convolve_naive(float* in, int input_length,
                   float* kernel,	int kernel_length, float* out)
{
    for (int i = 0; i < input_length + kernel_length - 1; i++) {
        out[i] = 0.0;
        int startk = i >= input_length ? i - input_length + 1 : 0;
        int endk = i < kernel_length ? i : kernel_length - 1;
        for (int k = startk; k <= endk; k++) {
            out[i] += in[i - k] * kernel[k];
        }
    }

    return 0;
}

std::vector<float> Algo::convolve(std::vector<float>& input, std::vector<float>& kernel, int mode){
    std::vector<float> output;
    int M = input.size();
    int N = kernel.size();
    output.resize(M + N - 1);
    convolve_sse(input.data(), M, kernel.data(), N, output.data());

    if(mode == 0) return output;
    else if(mode == 1){
        output.erase(output.begin(), output.begin() + N-1);
        for(int i=0; i<N-1; i++) output.pop_back();
        return output;
    }

    return output;
}

float Algo::squaredSum(std::vector<float>& x){
    float sum = 0;
    for(auto& item : x) sum += pow(item,2);
    return sum;
}

std::vector<float> Algo::arange(float start, float stop, float step, bool include) {
    std::vector<float> values;
    for (float value = start; value < stop; value += step)
        values.push_back(value);
    if(include) values.push_back(stop);
    return values;
}

void Algo::setCol(Eigen::MatrixXf& matrix, int colnum, float val){
    for(int i=0; i<matrix.rows(); i++){
        matrix(i, colnum) = val;
    }
}

void Algo::eigenXSort(Eigen::MatrixXf& matrix)
{
    std::vector<Eigen::VectorXf> vec;
    for (int64_t i = 0; i < matrix.rows(); ++i)
        vec.push_back(matrix.row(i));
    std::sort(vec.begin(), vec.end(), [](Eigen::VectorXf const& t1, Eigen::VectorXf const& t2){ return t1(0) < t2(0); } );
    for (int64_t i = 0; i < matrix.rows(); ++i)
        matrix.row(i) = vec[i];
};

std::vector<int> Algo::eigenAngleSort(Eigen::MatrixXf& matrix)
{
    std::vector<int> idx(matrix.rows());
    iota(idx.begin(), idx.end(), 0);
    std::vector<float> angs;
    for (int64_t i = 0; i < matrix.rows(); ++i) {
        float angle = -((atan2f(matrix(i,1), matrix(i,0)) * 180 / M_PI) - 90);
        angs.emplace_back(angle);
    }
    std::sort(idx.begin(), idx.end(),
              [&angs](int i1, int i2) {return angs[i1] < angs[i2];});
    Eigen::MatrixXf matrix_copy = matrix;
    for(int i=0; i<idx.size(); i++){
        matrix.row(i) = matrix_copy.row(idx[i]);
    }
    return idx;
};

std::vector<std::vector<int>> Algo::splitVector(const std::vector<int>& idx, int split_count){
    int each_contains = (int)(((float)idx.size() / (float)split_count));
    int remainder = idx.size() % split_count;
    std::vector<std::vector<int>> splits;
    for(int j=0; j<split_count; j++){
        if(j == split_count - 1)
            splits.emplace_back(std::vector<int>(idx.begin() + j*each_contains, idx.begin() + j*each_contains + each_contains + remainder));
        else
            splits.emplace_back(std::vector<int>(idx.begin() + j*each_contains, idx.begin() + j*each_contains + each_contains));
    }
    return splits;
}

//template< class T >
//void Algo::reorder(std::vector<T>& vA, const std::vector<size_t>& vOrder)
//{
//    assert(vA.size() == vOrder.size());
//    std::vector<T> vCopy = vA; // Can we avoid this?
//    for(int i = 0; i < vOrder.size(); ++i)
//        vA[i] = vCopy[ vOrder[i] ];
//}

std::vector<std::vector<std::vector<int>>> Algo::generateContPerms(int index_count, int split_count){
    struct custom_sort
    {
        inline bool operator() (const std::vector<int>& struct1, const std::vector<int>& struct2)
        {
            return (struct1[0] < struct2[0]);
        }
    };
    // Generate all continious permutations
    std::vector<int> idx(index_count);
    iota(idx.begin(), idx.end(), 0);
    std::vector<std::vector<int>> permutations;
    do {
        permutations.emplace_back(idx);
    } while (std::next_permutation(idx.begin(), idx.end()));
    std::vector<std::vector<std::vector<int>>> contiguous_perms;
    std::map<std::string, int> hash;
    for(const auto& perm : permutations){
        std::vector<std::vector<int>> splits = splitVector(perm, split_count);
        // Check if each split is continuous
        bool sorted = true;
        for(const auto& split : splits){
            if(!std::is_sorted(split.begin(), split.end())) sorted = false;
        }
        // Sort my splits
        std::sort(splits.begin(), splits.end(), custom_sort());
        if(sorted) {
            // Generate Key
            std::string s = "";
            for(const auto& split : splits){
                for(const auto& v : split) s+=std::to_string(v);
                s+="-";
            }
            if(hash.count(s)) continue;
            else hash[s] = 1;
            contiguous_perms.emplace_back(splits);
        }
    }
    return contiguous_perms;
}

void Algo::removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

Eigen::MatrixXf Algo::customSort(const Eigen::MatrixXf& matrix, const std::vector<int>& custom_order){
    if(matrix.rows() == custom_order.size()) {
        Eigen::MatrixXf matrix_copy = matrix;
        for (int i = 0; i < custom_order.size(); i++) {
            matrix_copy.row(i) = matrix.row(custom_order[i]);
        }
        return matrix_copy;
    }else if(matrix.rows() > custom_order.size()){
        Eigen::MatrixXf matrix_copy;
        matrix_copy.resize(custom_order.size(), matrix.cols());
        for (int i = 0; i < custom_order.size(); i++) {
            matrix_copy.row(i) = matrix.row(custom_order[i]);
        }
        return matrix_copy;
    }else{
        throw std::runtime_error("custom_sort error");
    }
}

void Algo::removeRows(Eigen::MatrixXf& matrix, std::vector<int>& to_remove){
    int count = 0;
    for(auto& row : to_remove){
        removeRow(matrix, row-count);
        count++;
    }
}

std::vector<float> Algo::linspace(float a, float b, size_t N) {
    float h = (b - a) / static_cast<float>(N-1);
    std::vector<float> xs(N);
    typename std::vector<float>::iterator x;
    float val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

std::vector<std::vector<std::vector<int>>> Algo::getPartitions(const std::vector<int>& elements){
    std::vector<std::vector<std::vector<int>>> fList;

    std::vector<std::vector<int>> lists;
    std::vector<int> indexes(elements.size(), 0); // Allocate?
    lists.emplace_back(std::vector<int>());
    lists[0].insert(lists[0].end(), elements.begin(), elements.end());

    int counter = -1;

    for(;;){
        counter += 1;
        fList.emplace_back(lists);

        int i,index;
        bool obreak = false;
        for (i=indexes.size()-1;; --i) {
            if (i<=0){
                obreak = true;
                break;
            }
            index = indexes[i];
            lists[index].erase(lists[index].begin() + lists[index].size()-1);
            if (lists[index].size()>0)
                break;
            lists.erase(lists.begin() + index);
        }
        if(obreak) break;

        ++index;
        if (index >= lists.size())
            lists.emplace_back(std::vector<int>());
        for (;i<indexes.size();++i) {
            indexes[i]=index;
            lists[index].emplace_back(elements[i]);
            index=0;
        }
    }

    struct csort
    {
        inline bool operator() (const std::vector<std::vector<int>>& struct1, const std::vector<std::vector<int>>& struct2)
        {
            return (struct1.size() < struct2.size());
        }
    };
    std::sort(fList.begin(), fList.end(), csort());

    return fList;
}

std::vector<std::vector<int>> Algo::getPermutations(const std::vector<int>& elements){
    std::vector<std::vector<int>> p;
    int n = elements.size();
    std::vector<int> elementsCpy = std::vector<int>(elements);
    int* myints = &elementsCpy[0];
    do {
        p.emplace_back(elementsCpy);
    } while ( std::next_permutation(myints,myints+n) );
    return p;
}

Eigen::MatrixXf Algo::fitBSpline(const Eigen::MatrixXf& input_pts_x, std::shared_ptr<SplineParamsVec>& splineParamsVec, bool computeCurve){
    int mode = 1;
    int hack = 1;
    std::vector<SplineParams>& allParams = splineParamsVec->splineParams;

    // Hack
    Eigen::MatrixXf input_pts = input_pts_x;

    // Copy
    Eigen::MatrixXf inputs = input_pts;

    if(hack){
        // Reverse Rows
        inputs = input_pts.colwise().reverse();

        // Add two control points at end
        Eigen::Vector3f c1 = Eigen::Vector3f((inputs(0,0)+inputs(1,0))/2.,(inputs(0,1)+inputs(1,1))/2.,(inputs(0,2)+inputs(1,2))/2.);
        int in = inputs.rows()-1;
        Eigen::Vector3f c2 = Eigen::Vector3f((inputs(in,0)+inputs(in-1,0))/2.,(inputs(in,1)+inputs(in-1,1))/2.,(inputs(in,2)+inputs(in-1,2))/2.);
        inputs.conservativeResize(inputs.rows()+2, inputs.cols());
        inputs.row(inputs.rows()-2) = c1;
        inputs.row(inputs.rows()-1) = c2;
    }

    // Initialize entry
    int n = inputs.rows() - 2;
    int n1 = n+1;
    std::vector<float> dx(n, 0);
    std::vector<float> dy(n, 0);

    // First and Last Derivs
    if(mode == 0){
        dx[0] = inputs(n, 0) - inputs(0, 0);
        dy[0] = inputs(n, 1) - inputs(0, 1);
        dx[n-1] = -(inputs(n1, 0) - inputs(n-1, 0));
        dy[n-1] = -(inputs(n1, 1) - inputs(n-1, 1));
    }else if(mode == 1){
        float DIV = 3.;
        dx[0] = (inputs(1, 0) - inputs(0, 0))/DIV;
        dy[0] = (inputs(1, 1) - inputs(0, 1))/DIV;
        dx[n-1] = (inputs(n-1, 0) - inputs(n-2, 0))/DIV;
        dy[n-1] = (inputs(n-1, 1) - inputs(n-2, 1))/DIV;
    }

    // Fill other control derivs
    std::vector<float> Ax(n, 0);
    std::vector<float> Ay(n, 0);
    std::vector<float> Bi(n, 0);
    Bi[1] = -1./inputs(1, 2);
    Ax[1] = -(inputs(2, 0) - inputs(0, 0) - dx[0])*Bi[1];
    Ay[1] = -(inputs(2, 1) - inputs(0, 1) - dy[0])*Bi[1];
    for(int i=2; i<n-1; i++){
        Bi[i] = -1/(inputs(i,2) + Bi[i-1]);
        Ax[i] = -(inputs(i+1,0) - inputs(i-1,0) - Ax[i-1])*Bi[i];
        Ay[i] = -(inputs(i+1,1) - inputs(i-1,1) - Ay[i-1])*Bi[i];
    }
    for(int i=n-2; i>0; i--){
        dx[i] = Ax[i] + dx[i+1]*Bi[i];
        dy[i] = Ay[i] + dy[i+1]*Bi[i];
    }

    // Interpolate
    std::vector<Eigen::Vector2f> paths = {Eigen::Vector2f(inputs(0,0), inputs(0,1))};
    for(int i=0; i<n-1; i++){
        // Distance
        float dist = sqrt(pow((inputs(i,0) - inputs(i+1,0)),2) + pow((inputs(i,1) - inputs(i+1,1)),2));
        float count = (float)((int)(dist/0.01));

        // P Vals
        SplineParams params;
        params.p0(0) = inputs(i,0); params.p0(1) = inputs(i,1);
        params.p1(0) = inputs(i,0) + dx[i]; params.p1(1) = inputs(i,1) + dy[i];
        params.p2(0) = inputs(i+1,0) - dx[i+1]; params.p2(1) = inputs(i+1,1) - dy[i+1];
        params.p3(0) = inputs(i+1,0); params.p3(1) = inputs(i+1,1);
        params.Ax = params.p3(0) - 3*params.p2(0) + 3*params.p1(0) - params.p0(0);
        params.Bx = 3*(params.p2(0) - 2*params.p1(0) + params.p0(0));
        params.Cx = 3*(params.p1(0) - params.p0(0));
        params.Dx = params.p0(0);
        params.Ay = params.p3(1) - 3*params.p2(1) + 3*params.p1(1) - params.p0(1);
        params.By = 3*(params.p2(1) - 2*params.p1(1) + params.p0(1));
        params.Cy = 3*(params.p1(1) - params.p0(1));
        params.Dy = params.p0(1);
        allParams.emplace_back(params);

        // Interpolate
        if(computeCurve) {
            float extend = 0.0;
            auto ts = linspace(0. - extend, 1 + extend - (1. / count), count);
            for (auto t: ts) {
                float X = t * t * t * (params.Ax) + t * t * (params.Bx) + t * params.Cx + params.Dx;
                float Y = t * t * t * (params.Ay) + t * t * (params.By) + t * params.Cy + params.Dy;
                paths.emplace_back(Eigen::Vector2f(X, Y));
            }
        }

        //http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html

        // ** Add to my notes. - https://arxiv.org/pdf/1904.08913.pdf
        // Uber Paper
        // We use the stereo image disparity as flow truth?

        // First step is to test if training worked check weights
        // Then work through the training example
        // Then rewrite from scratch own module to train with just D Net and R Net no K
        // Setup framework to evaluate the performance of depth estimation

        // Load a single instance of kitti and pose, validate stereo and temporal transforms.
        // Take note we have stereo drive. Try visualize both kitti and stereo drive data see quality of pointcloid
    }

    // Special Case for 1 item
    if(paths.size() == 1){
        Eigen::Vector2f lVec = paths[0];
        Eigen::Vector2f rVec = paths[0];
        lVec(0) += 0.2;
        rVec(0) -= 0.2;
        paths.emplace_back(lVec);
        paths.emplace_back(rVec);
    }

    if(hack){
        std::reverse(paths.begin(), paths.end());
    }

    // Convert to eigen
    Eigen::MatrixXf outputs;
    outputs.resize(paths.size(),2);
    for(int i=0; i<paths.size(); i++){
        outputs(i,0) = paths[i](0);
        outputs(i,1) = paths[i](1);
    }

    return outputs;
}

Eigen::MatrixXf Algo::solveT(const std::shared_ptr<SplineParamsVec>& splineParamsVec, Eigen::MatrixXf inputPts){
    /*
     * We want to solve for the closest Pt
     */
    //std::cout << allParams.size() << std::endl;
    //std::cout << allParams[0].p0 << std::endl;
    const std::vector<SplineParams>& allParams = splineParamsVec->splineParams;
    std::vector<Eigen::Vector3f> projPoints;

    // Iterate each point
    auto begin = std::chrono::steady_clock::now();
    for(int i=0; i<inputPts.rows(); i++){
        float Mx = inputPts(i,0);
        float My = inputPts(i,1);

        Eigen::Vector3f closestPoint;
        float closestDistance = std::numeric_limits<float>::infinity();
        bool closestFound = false;

        for(int j=0; j<allParams.size(); j++){
            const auto& params = allParams[j];
            float Ax = params.Ax;
            float Bx = params.Bx;
            float Cx = params.Cx;
            float Dx = params.Dx;
            float Ay = params.Ay;
            float By = params.By;
            float Cy = params.Cy;
            float Dy = params.Dy;

            float A = 6*Ax*Ax + 6*Ay*Ay; // t5
            float B = 10*Ax*Bx + 10*Ay*By; // t4
            float C = 8*Ax*Cx + 8*Ay*Cy + 4*Bx*Bx + 4*By*By; // t3
            float D = 6*Ax*Dx - 6*Ax*Mx + 6*Ay*Dy - 6*Ay*My + 6*Bx*Cx + 6*By*Cy; // t2
            float E = 4*Bx*Dx - 4*Bx*Mx + 2*Cx*Cx + 4*By*Dy - 4*By*My + 2*Cy*Cy; // t1
            float F = 2*Cx*Dx - 2*Cx*Mx + 2*Cy*Dy - 2*Cy*My; // 1

            std::vector<double> sln(5);
            int result = Poly34::SolveP5(sln.data(), B/A, C/A, D/A, E/A, F/A);
            std::vector<float> ts;
            float tLimitS = 0.0; float tLimitE = 1.0;

            // Should i just extend to infinity?
            if(j == 0) tLimitS = -0.3;
            if(j == allParams.size() - 1) tLimitE = 1.3;

            if(result == 5){
                for(auto r : sln){
                    if(r >= tLimitS && r <= tLimitE)
                        ts.emplace_back(r);
                }
            }else if(result == 3){
                for(int k=0; k<3; k++){
                    if(sln[k] >= tLimitS && sln[k] <= tLimitE)
                        ts.emplace_back(sln[k]);
                }
            }else if(result == 1){
                if(sln[0] >= tLimitS && sln[0] <= tLimitE)
                    ts.emplace_back(sln[0]);
            }

            // Forward compute
            for(auto t : ts){
                float X = t*t*t*(Ax) + t*t*(Bx) + t*(Cx) + Dx;
                float Y = t*t*t*(Ay) + t*t*(By) + t*(Cy) + Dy;
                float l2Dist = sqrt(pow(X-Mx,2)+pow(Y-My,2));
                if(l2Dist < closestDistance){
                    closestDistance = l2Dist;
                    closestPoint = Eigen::Vector3f(X,Y,l2Dist);
                    closestFound = true;
                }
            }

        }

        if(closestFound)
            projPoints.emplace_back(closestPoint);
    }

    //std::cout << "solveT = " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count() << "[us]" << std::endl;

    // Convert to eigen
    Eigen::MatrixXf outputs;
    outputs.resize(projPoints.size(),3);
    for(int i=0; i<projPoints.size(); i++){
        outputs(i,0) = projPoints[i](0);
        outputs(i,1) = projPoints[i](1);
        outputs(i,2) = projPoints[i](2);
    }
    return outputs;
}

Eigen::VectorXf Algo::gaussian(const Eigen::VectorXf& x, float mu, float sig, float power){
    Eigen::VectorXf a = Eigen::exp(-Eigen::pow(Eigen::abs(x.array() - mu), power) / (2 * pow(sig, power)));
    float maxv = a.maxCoeff();
    float minv = a.minCoeff();
    a = (a.array() - minv)/(maxv - minv);
    return a;
}

std::vector<cv::Point2f> getImageCoordinates(int imgh, int imgw) {
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

std::vector<float> Algo::generateCameraAngles(const cv::Mat& K, const cv::Mat& d, int imgw, int imgh){
    std::vector<cv::Point2f> img_coords;
    std::vector<cv::Point2f> norm_coords(imgh * imgw);
    img_coords = getImageCoordinates(imgh, imgw);

    cv::Mat nmap;
    cv::Mat midrays[3];
    nmap = cv::Mat(imgh, imgw, CV_32FC3);

    cv::Mat KK_;//(KK().rows(), KK().cols(), CV_32FC1, KK().data());
    cv::Mat Kd_;//(Kd().rows(), Kd().cols(), CV_32FC1, Kd().data());
    KK_ = K;
    Kd_ = d;

    cv::undistortPoints(img_coords, norm_coords, KK_, Kd_);

    for (int i = 0; i < imgw; i++)
    {
        for (int j = 0; j < imgh; j++)
        {
            int pix_num = i*imgh+j;
            float x_dir = norm_coords[pix_num].x;
            float y_dir = norm_coords[pix_num].y;
            float z_dir = 1.0f;
            float mag = sqrt(x_dir*x_dir + y_dir*y_dir + z_dir*z_dir);
            cv::Vec3f pix_ray = cv::Vec3f(x_dir/mag,y_dir/mag,z_dir/mag);
            cv::Vec3f pix_ray_nn = cv::Vec3f(x_dir,y_dir,z_dir);
            nmap.at<cv::Vec3f>(j,i) = pix_ray;
        }
    }

    // Split midrays into 3 channels, x(0), y(1), z(2)
    split(nmap.row(imgh/2-1).clone(),midrays);

    // Compute Valid Angles (If we compute design points with these theta, they are guaranteed to lie on ray)
    std::vector<float> valid_angles;
    for(int i=0; i<midrays[0].size().width; i++){
        float x = midrays[0].at<float>(0,i);
        float y = midrays[1].at<float>(0,i);
        float z = midrays[2].at<float>(0,i);
        float theta = -((atan2f(z, x) * 180 / M_PI) - 90);
        valid_angles.emplace_back(theta);
    }

    return valid_angles;
}