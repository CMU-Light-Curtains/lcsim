#include "fitting.h"

using namespace lc;

Fitting::Fitting(std::shared_ptr<DatumProcessor> datumProcessor){
    datumProcessor_ = datumProcessor;
}

Fitting::~Fitting(){
}

std::vector<std::vector<Point2D>> Fitting::segmentClusters(const std::vector<Point2D>& design_pts, const std::vector<Point2D>& pts){
    std::vector<Point2D> remains;

    std::vector<std::vector<Point2D>> clusters;
    std::vector<Point2D> curr_cluster, temp_cluster;

    const float l2threshold = 0.01;
    const int nothingFoundThreshold = 100;
    const int clusterMinSize = 2;

    // Iterate pts
    bool first_run = true;
    int nothingFoundCounter = 0;
    int i = -1;
    for(const auto& pt : pts){
        i++;

        // Add pt to curr cluster
        temp_cluster.emplace_back(pt);

        // Check if reached a design pt
        auto closest_pt = Algo::nn(pt, design_pts);
        auto distance = sqrt(pow(pt(0)-closest_pt(0),2) + pow(pt(1)-closest_pt(1),2));
        if(distance < l2threshold){

            // Below Threshold, Update Curr Cluster
            if(nothingFoundCounter < nothingFoundThreshold){
                curr_cluster.insert(curr_cluster.end(), temp_cluster.begin(), temp_cluster.end());
                temp_cluster.clear();
            }
                // Above Threshold
            else{
                if(curr_cluster.size() >= clusterMinSize) clusters.emplace_back(std::vector<Point2D>(curr_cluster));
                if(temp_cluster.size() >= clusterMinSize) clusters.emplace_back(std::vector<Point2D>(temp_cluster));
                curr_cluster.clear();
                temp_cluster.clear();
                //std::cout << "******" << std::endl;
                //std::cout << clusters.back().back() << std::endl;
            }

            nothingFoundCounter = 0;
        }else{
            nothingFoundCounter++;
        }

        // Last one
        if(i == pts.size() - 1){
            if(nothingFoundCounter < nothingFoundThreshold){
                if(curr_cluster.size() >= clusterMinSize) clusters.emplace_back(std::vector<Point2D>(curr_cluster));
            }
            else{
                if(temp_cluster.size() >= clusterMinSize) clusters.emplace_back(std::vector<Point2D>(temp_cluster));
                if(curr_cluster.size() >= clusterMinSize) clusters.emplace_back(std::vector<Point2D>(curr_cluster));
            }
        }


    }

    return clusters;
}

std::vector<std::vector<Point2D>> Fitting::listSort(const std::vector<std::vector<Point2D>>& list){
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
    std::vector<std::vector<int>> permutations = Algo::getPermutations(indexes);

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

std::vector<Eigen::MatrixXf> Fitting::curtainSplitting(Eigen::MatrixXf& spline, std::string cam_name, std::string laser_name){
    // Get Objects
    const Datum& cam_data = *(datumProcessor_->getCDatum(cam_name).get());
    const Laser& laser_data = cam_data.laser_data.at(laser_name);

    // Convert to Vec
    std::vector<Point2D> pts(spline.rows());
    for(int i=0; i<pts.size(); i++) pts[i] = Point2D(spline(i, 0),spline(i, 1));

    // Remove everything that is not good index
    auto good_inds = datumProcessor_->checkPoints(pts, cam_data, laser_data);
    std::vector<Point2D> pts_good(good_inds.size());
    for(int i=0; i<pts_good.size(); i++) {
        pts_good[i] = pts[good_inds[i]];
    }
    pts = pts_good;
    good_inds.clear();

    // Iterate and Split
    std::vector<Eigen::MatrixXf> endProduct;
    std::vector<std::vector<Point2D>> consComponents;
    std::vector<std::vector<Point2D>> currComponents = {pts};
    const int maxIter = 100;
    for(int m=0; m<maxIter; m++){
        if(m == maxIter - 1) throw std::runtime_error("Max iterations reached");
        if(currComponents.size() == 0){
            break;
        }

        // Iterate currComponents
        std::vector<std::vector<Point2D>> newCurrComponents;
        for(auto& component : currComponents){
            // Compute Design Pts
            auto design_pts = datumProcessor_->findCameraIntersectionsOpt(cam_data, good_inds, component);

            // Convert to 2D form
            std::vector<Point2D> design_pts_2d;
            for(int i=0; i<design_pts.cols(); i++){
                if(design_pts(3,i) == -1) continue;
                design_pts_2d.emplace_back(Point2D(design_pts(0,i), design_pts(2,i)));
            }

            // Find the remains
            auto clusters = segmentClusters(design_pts_2d, component);

            // Add to vec
            if(clusters.size() == 1){
                consComponents.emplace_back(clusters[0]);
            }else{
                newCurrComponents.insert(newCurrComponents.end(), clusters.begin(), clusters.end());
            }
        }

        currComponents = newCurrComponents;
    }

    std::cout << "--" << std::endl;

    // Generate Partitions
    int consComponentSize = consComponents.size();
    std::vector<int> elements;
    for(int i=0; i<consComponentSize; i++){
        elements.emplace_back(i);
    }
    auto fListsX = Algo::getPartitions(elements);
    std::map<int, std::vector<std::vector<std::vector<int>>>> partitionsMap;
    for(int i=1; i<=elements.size(); i++){
        partitionsMap[i] = std::vector<std::vector<std::vector<int>>>();
    }
    for(auto& lists : fListsX){
        partitionsMap[lists.size()].emplace_back(lists);
    }

    // Iterate partitions
    bool debug = false;
    std::vector<std::vector<int>> bestLists;
    float lowestCost = std::numeric_limits<float>::infinity();
    for (auto const& partitions : partitionsMap)
    {
        int curtainSize = partitions.first;
        const auto& fLists = partitions.second;

        std::cout << "*Count* " << curtainSize << std::endl;

        // Generate the set of points
        bestLists.clear();
        lowestCost = std::numeric_limits<float>::infinity();
        std::vector<std::pair<int, float>> costs(fLists.size(), std::pair<int,float>(0,lowestCost));
        #pragma omp parallel for shared(fLists, costs, consComponents)
        for(int ind=0; ind<fLists.size(); ind++){
            //std::cout << (float)ind/(float)fLists.size() << " " << curtainSize << std::endl;
            auto const& lists = fLists[ind];

            // Debug
            if(debug) {
                std::cout << "-[";
                for (auto &l : lists) {
                    std::cout << "(";
                    for (auto &e : l) {
                        std::cout << e << " ";
                    }
                    std::cout << ") ";
                }
                std::cout << "]-" << std::endl;
            }

            //(0 1 ) (2 )
            float totalCost = 0;
            for(auto& l : lists){
                float currCost = 0;

                // Debug
                if(debug){
                    std::cout << "    (";
                    for(auto& e : l){
                        std::cout << e << " ";
                    }
                    std::cout << ") " << std::endl;
                }

                // Sort based on distance metric to combine

//                    // (0 1 )
//                    std::vector<Point2D> combinedPts;
//                    for(auto& e : l){
//                        auto pts = consComponents[e];
//                        combinedPts.insert(combinedPts.end(), pts.begin(), pts.end());
//                    }
//                    //angleSort(combinedPts);
                std::vector<Point2D> combinedPts;
                std::vector<std::vector<Point2D>> combinedPtsVec;
                for (auto &e : l) {
                    auto pts = consComponents[e];
                    combinedPtsVec.emplace_back(pts);
                }
                combinedPtsVec = listSort(combinedPtsVec);
                for(auto & e : combinedPtsVec){
                    combinedPts.insert(combinedPts.end(), e.begin(), e.end());
                }

                // Compute Design Pts
                auto design_pts = datumProcessor_->findCameraIntersectionsOpt(cam_data, good_inds, combinedPts);
                std::vector<Point2D> design_pts_2d;
                for(int i=0; i<design_pts.cols(); i++){
                    if(design_pts(3,i) == -1) continue;
                    design_pts_2d.emplace_back(Point2D(design_pts(0,i), design_pts(2,i)));
                }

                // Need to check if we got all points
                auto clusters = segmentClusters(design_pts_2d, combinedPts);
                if(clusters.size() != 1) {
                    if(debug) std::cout << "    clustersize: " << clusters.size() << std::endl;
                    currCost = std::numeric_limits<float>::infinity();
                }

                // Get Velo and Accel
                std::shared_ptr<Angles> angles_ptr = datumProcessor_->calculateAngles(design_pts, cam_data, laser_data, false, false, false);
                Angles& angles = *angles_ptr.get();

                // Calculate cost
                Algo::removeNan(angles.velocities);
                Algo::removeNan(angles.accels);
                if(angles.accels.size() < 10){
                    if(debug) std::cout << "    asize: " << angles.accels.size() << std::endl;
                    currCost = std::numeric_limits<float>::infinity();
                    totalCost += currCost;
                    continue;
                }
                std::vector<float> smoothing_kernel = {0.2, 0.2, 0.2, 0.2, 0.2};
                std::vector<float> edge_kernel = {-1, -2, 0, 1, 2};
                angles.accels = Algo::convolve(angles.accels, smoothing_kernel, 1);
                auto jerk = Algo::convolve(angles.accels, edge_kernel, 1);
                angles.summed_peak = Algo::squaredSum(jerk);
                //if(angles.max_velo > 25000) currCost = std::numeric_limits<float>::infinity();
                float max_velo = *std::max_element(angles.velocities.begin(), angles.velocities.end(), [](const float& a, const float& b) { return abs(a) < abs(b); });
                float max_accel = *std::max_element(angles.accels.begin(), angles.accels.end(), [](const float& a, const float& b) { return abs(a) < abs(b); });
                if(max_velo > 25000) {
                    if(debug) std::cout << "    max_velo: " << max_velo << std::endl;
                    currCost = std::numeric_limits<float>::infinity();
                }
                if(max_accel > 5e7) {
                    if(debug) std::cout << "    max_accel: " << max_accel << std::endl;
                    currCost = std::numeric_limits<float>::infinity();
                }
                currCost += angles.summed_peak;

                // Add cost
                totalCost += currCost;
                if(debug) std::cout << "    " << currCost << std::endl;
            }

            // Set cost
            costs[ind] = std::pair<int, float>(ind, totalCost);
        }

        // Sort and extract
        struct csort
        {
            inline bool operator() (const std::pair<int, float>& struct1, const std::pair<int, float>& struct2)
            {
                return (struct1.second < struct2.second);
            }
        };
        std::sort(costs.begin(), costs.end(), csort());
        lowestCost = costs[0].second;
        bestLists = fLists[costs[0].first];


        // Check if something valid found
        if(!std::isinf(lowestCost)){
            // Debug
            if(debug){
                std::cout << curtainSize << std::endl;
                std::cout << "-[";
                for(auto& l : bestLists){
                    std::cout << "(";
                    for(auto& e : l){
                        std::cout << e << " ";
                    }
                    std::cout << ") ";
                }
                std::cout << "]-" << std::endl;

            }
            std::cout << "Found" << std::endl;
            break;
        }

    }

    // Force return a joint spline
    //bestLists = {{1,2,3}};

    std::vector<std::vector<Point2D>> final;
    for(auto& l : bestLists) {

//            // (0 1 )
//            std::vector<Point2D> combinedPts;
//            for (auto &e : l) {
//                auto pts = consComponents[e];
//                combinedPts.insert(combinedPts.end(), pts.begin(), pts.end());
//            }
//            //angleSort(combinedPts);

        std::vector<Point2D> combinedPts;
        std::vector<std::vector<Point2D>> combinedPtsVec;
        for (auto &e : l) {
            auto pts = consComponents[e];
            combinedPtsVec.emplace_back(pts);
        }
        combinedPtsVec = listSort(combinedPtsVec);
        for(auto & e : combinedPtsVec){
            combinedPts.insert(combinedPts.end(), e.begin(), e.end());
        }

        // Compute Design Pts
        auto design_pts = datumProcessor_->findCameraIntersectionsOpt(cam_data, good_inds, combinedPts);
        std::vector<Point2D> design_pts_2d;
        for (int i = 0; i < design_pts.cols(); i++) {
            if (design_pts(3, i) == -1) continue;
            design_pts_2d.emplace_back(Point2D(design_pts(0, i), design_pts(2, i)));
        }

        //final.emplace_back(combinedPts);
        final.emplace_back(design_pts_2d);
    }

    // Convert and return cluster design pts
    for(auto& cluster : final){
        Eigen::MatrixXf cluster_eigen; cluster_eigen.resize(cluster.size(),2);
        for(int i=0; i<cluster.size(); i++){
            cluster_eigen(i,0) = cluster[i](0);
            cluster_eigen(i,1) = cluster[i](1);
        }
        endProduct.emplace_back(cluster_eigen);
    }
    return endProduct;


//        // Convert and return original points
//        for(auto& cluster : consComponents){
//            Eigen::MatrixXf cluster_eigen; cluster_eigen.resize(cluster.size(),2);
//            for(int i=0; i<cluster.size(); i++){
//                cluster_eigen(i,0) = cluster[i](0);
//                cluster_eigen(i,1) = cluster[i](1);
//            }
//            endProduct.emplace_back(cluster_eigen);
//        }
//        return endProduct;

    //exit(-1);

//        // Convert and return cluster design pts
//        for(auto& cluster : consComponents){
//
//            auto design_pts = datumProcessor_->findCameraIntersectionsOpt(cam_data, good_inds, cluster);
//            std::vector<Point2D> design_pts_2d;
//            for(int i=0; i<design_pts.cols(); i++){
//                if(design_pts(3,i) == -1) continue;
//                design_pts_2d.emplace_back(Point2D(design_pts(0,i), design_pts(2,i)));
//            }
//
//            Eigen::MatrixXf cluster_eigen; cluster_eigen.resize(design_pts_2d.size(),2);
//            for(int i=0; i<design_pts_2d.size(); i++){
//                cluster_eigen(i,0) = design_pts_2d[i](0);
//                cluster_eigen(i,1) = design_pts_2d[i](1);
//            }
//            endProduct.emplace_back(cluster_eigen);
//        }
//        return endProduct;
}

std::shared_ptr<Angles> Fitting::splineToAngles(Eigen::MatrixXf& spline, std::string cam_name, std::string laser_name, bool vlimit){
    // Get Objects
    const Datum& cam_data = *(datumProcessor_->getCDatum(cam_name).get());
    const Laser& laser_data = cam_data.laser_data.at(laser_name);

    // Convert to vec
    std::vector<Point2D> pts(spline.rows());
    for(int i=0; i<pts.size(); i++) pts[i] = Point2D(spline(i, 0),spline(i, 1));

    // Compute Angles/Velo/Accel
    auto good_inds = datumProcessor_->checkPoints(pts, cam_data, laser_data);
    auto begin = std::chrono::steady_clock::now();
    auto design_pts = datumProcessor_->findCameraIntersectionsOpt(cam_data, good_inds, pts);
    //auto design_pts = findCameraIntersectionsOpt2(cam_data, good_inds, pts);
    //for(int i=0; i<design_pts.cols(); i++){
    //    std::cout << design_pts(0,i) << " " << design_pts(2,i) << std::endl;
    //    std::cout << design_pts2(0,i) << " " << design_pts2(2,i) << std::endl;
    //}
    //std::cout << "findCameraIntersectionsOpt = " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count() << "[us]" << std::endl;
    std::shared_ptr<Angles> angles_ptr = datumProcessor_->calculateAngles(design_pts, cam_data, laser_data, true, false, vlimit);
    Angles& angles = *angles_ptr.get();

    // Smooth and get peaks
    Algo::removeNan(angles.velocities);
    Algo::removeNan(angles.accels);
    if(angles.accels.size() < 11){
        return angles_ptr;
    }
    std::vector<float> smoothing_kernel = {0.2, 0.2, 0.2, 0.2, 0.2};
    std::vector<float> edge_kernel = {-1, -2, 0, 1, 2};
    //std::vector<float> edge_kernel = {-4, +2, 0, -2, 4};
    angles.accels = Algo::convolve(angles.accels, smoothing_kernel, 1);
    angles.peaks = Algo::convolve(angles.accels, edge_kernel, 1);
    angles.summed_peak = Algo::squaredSum(angles.peaks);
    angles.max_velo = *std::max_element(angles.velocities.begin(), angles.velocities.end()); // slow. move this out to the calculateAnglesFunc
    angles.design_pts = design_pts;

    return angles_ptr;
}

std::pair<Eigen::MatrixXf, float> Fitting::fitSpline(Eigen::MatrixXf& path, std::string cam_name, std::string laser_name){
    auto begin = std::chrono::steady_clock::now();
    float best_b = 0;
    std::tuple<Eigen::MatrixXf, float, bool> best_data;

    // Create copy
    Eigen::MatrixXf path_copy = path;
    std::shared_ptr<SplineParamsVec> allParams = std::make_shared<SplineParamsVec>(SplineParamsVec());

    // Special Cases
    if(path.rows() == 1){
        Eigen::MatrixXf spline = Algo::fitBSpline(path_copy,allParams);
        return std::pair<Eigen::MatrixXf, float>(spline, 0);
    }else if(path.rows() == 2){
        float cost = 0;
        Eigen::MatrixXf spline = Algo::fitBSpline(path_copy,allParams);
        std::shared_ptr<Angles> angles = splineToAngles(spline, cam_name, laser_name);
        // Compute distance
        Eigen::MatrixXf output_pts = angles->output_pts.transpose();
        bool exceed_dist = Algo::closestDistance(output_pts, path_copy, 0.1);
        // The points are no longer reaching, so we bias this badly
        if(exceed_dist){
            return std::pair<Eigen::MatrixXf, float>(spline, -1);
        }
        float delt = 0.01;
        cost += (1-delt)*angles->summed_peak + delt*angles->max_velo;
        return std::pair<Eigen::MatrixXf, float>(spline, cost);
    }

    begin = std::chrono::steady_clock::now();

    // Test annealing
    float start = 1.8;
    float end = 11.5;
    float step = 2;
    int counter = 0;
    std::map<float, float> hash1;
    std::map<float, Eigen::MatrixXf> hash2;
    std::map<float, bool> hash3;
    Eigen::MatrixXf best_spline;
    bool best_invalid;
    float best_cost;
    while(1){
        counter+=1;

        // Test set
        float lowest_cost = std::numeric_limits<float>::infinity();
        float curr_best_b = 0;
        bool curr_invalid = false;
        Eigen::MatrixXf curr_best_spline;
        for(auto b : Algo::arange(start, end, step, true)){
            //b = 11.5; //HACK!!!!!!!!!!!
            float cost = 0;
            Eigen::MatrixXf spline;
            bool invalid = false;
            if(hash1.count(b)){
                cost = hash1[b];
                spline = hash2[b];
                invalid = hash3[b];
            }else{
                Algo::setCol(path_copy, 2, b);
                spline = Algo::fitBSpline(path_copy, allParams);
                std::shared_ptr<Angles> angles = splineToAngles(spline, cam_name, laser_name);
                //Angles angles;
                //if(angles.exceed) invalid = true;

                // Compute distance
                Eigen::MatrixXf output_pts = angles->output_pts.transpose();
                bool exceed_dist = Algo::closestDistance(output_pts, path_copy, 0.1);
                // The points are no longer reaching, so we bias this badly
                if(exceed_dist){
                    invalid = true;
                    cost += 1000000000;
                }
                float delt = 0.01;
                cost += (1-delt)*angles->summed_peak + delt*angles->max_velo;
            }

            //std::cout << b << " " << cost << std::endl;

            // Cost Function
            hash1[b] = cost;
            hash2[b] = spline;
            hash3[b] = invalid;
            if(cost < lowest_cost){
                lowest_cost = cost;
                curr_best_spline = spline;
                curr_best_b = b;
                curr_invalid = invalid;
            }

        }

        // Update
        start = curr_best_b - step;
        end = curr_best_b + step;
        start = std::max(start, (float)1.8);
        end = std::min(end, (float)11.5);
        step /= 2.5;
        if(counter == 4){
            best_cost = lowest_cost;
            best_invalid = curr_invalid;
            best_b = curr_best_b;
            best_spline = curr_best_spline;
            break;
        }
    }

    //std::cout << " " << best_b << " " << best_cost << " " << std::endl;
    if(best_invalid){
        //ROS_ERROR("Invalid");
        best_cost = -1;
    }

//        //
//        Angles angles = splineToAngles(best_spline, cam_name, laser_name);
//        Eigen::MatrixXf design_pts = angles.design_pts.transpose();
//        float dist_sum = closestDistance(design_pts, path_copy, true);
//        std::cout << dist_sum << std::endl;

    //std::cout << "fitSpline = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;

    //exit(-1);

    return std::pair<Eigen::MatrixXf, float>(best_spline, best_cost);
}

void Fitting::curtainNodes(Eigen::MatrixXf& path, std::string cam_name, std::string laser_name, std::shared_ptr<Output>& output, bool process){
    /*
     * This function takes in just path (a single path)
     * sorts them xwise left to right
     *
     * I compute the spline -
     *  fitSpline() - does the optimization via annealing - return best spline
     *      this will call testSpline() - this does all the angles/gradient compute and returns it in Output object
     *
     * Need a cost for checking if the target actually got sampled
     */
    const Datum& cam_data = *(datumProcessor_->getCDatum(cam_name).get());
    const Laser& laser_data = cam_data.laser_data.at(laser_name);

    // Path remove the out of fov points
    std::vector<Point2D> pts(path.rows());
    for(int i=0; i<pts.size(); i++) pts[i] = Point2D(path(i, 0),path(i, 1));
    auto bad_inds = datumProcessor_->checkPoints(pts, cam_data, laser_data, false);
    Algo::removeRows(path, bad_inds);
    if(path.rows() == 0){
        return;
    }

    std::vector<Eigen::MatrixXf> finalSplines;

    // Start with angle sort for all
    Eigen::MatrixXf p1 = path;
    Algo::eigenAngleSort(p1);
    std::pair<Eigen::MatrixXf, float> s1 = fitSpline(p1, cam_name, laser_name);
    if(s1.second >= 0) {
        finalSplines.emplace_back(s1.first);
    }

    // If that failed do xsort
    if(finalSplines.empty()){
        Eigen::MatrixXf p2 = path;
        Algo::eigenXSort(p2);
        std::pair<Eigen::MatrixXf, float> s2 = fitSpline(p2, cam_name, laser_name);
        if(s2.second >= 0) {
            finalSplines.emplace_back(s2.first);
        }
    }

    // Plan N paths
    if(finalSplines.empty()){

        // Sort by angles again for all
        Algo::eigenAngleSort(path);

        //TEST INCREASING SPLIT COUNT
        for(int sc=2; sc<5; sc++){
            // Generate all continious permutations
            auto contiguous_perms = Algo::generateContPerms(path.rows(), sc);
            //std::cout << contiguous_perms.size() << std::endl;

            // We could sort the permutations based on average change in angle?
            auto beginx = std::chrono::steady_clock::now();
            std::vector<std::pair<int, float>> costs(contiguous_perms.size());
            for(int i=0; i<contiguous_perms.size(); i++) {
                const auto &splits = contiguous_perms[i];
                float ychange = 0.;
                float ycount = 0.;
                for(const auto& split : splits){
                    if(split.size() > 1) {
                        for (int j = 1; j < split.size(); j++) {
                            int rindex = split[j];
                            int lindex = split[j - 1];
                            ychange += fabs(path(rindex, 1) - path(lindex, 1));
                            ycount += 1.;
                        }
                    }else{
                        ycount += 1.;
                    }
                }
                float yavg = ychange/ycount;
                costs[i] = std::pair<int, float>(i, yavg);
            }
            // Sort
            std::sort(costs.begin(), costs.end(),
                      [](const std::pair<int, float>& c1, const std::pair<int, float>& c2) {return c1.second < c2.second;});
            // Reorganize order of perms
            std::vector<size_t> indicies(costs.size());
            for(int i=0; i<indicies.size(); i++) indicies[i] = costs[i].first;
            Algo::reorder(contiguous_perms, indicies);

            // Iterate and compute costs to break
            auto begin = std::chrono::steady_clock::now();
            bool added = false;
            int windex = -1;
            for(int i=0; i<contiguous_perms.size(); i++){
                const auto& splits = contiguous_perms[i];
                bool valid = true;
                float hit_percentage = ((float)i/(float)contiguous_perms.size())*100.;
                std::vector<Eigen::MatrixXf> goodSplines;
                for(auto& split : splits){
                    // Generate path
                    Eigen::MatrixXf split_path = Algo::customSort(path, split);
                    // Compute
                    std::pair<Eigen::MatrixXf, float> s = fitSpline(split_path, cam_name, laser_name);
                    if(s.second < 0) valid = false;
                    else goodSplines.emplace_back(s.first); // hack
                }
                if(valid){
                    finalSplines.insert(finalSplines.end(), goodSplines.begin(), goodSplines.end());
                    added = true;
                    windex = i;
                    break;
                }
                if(hit_percentage > 0.3) break; // Hack to make it faster for more splits
            }
            //std::cout << "split = " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "[ms]" << std::endl;
            if(added) break;
        }

    }

    // Allocate Outputs
    if(process) output->output_pts_set.resize(finalSplines.size());
    output->spline_set.resize(finalSplines.size());

    // Reprocess
    for(int i=0; i<finalSplines.size(); i++){
        std::shared_ptr<Output> temp_output = std::make_shared<Output>();
        if(process){
            datumProcessor_->processTest(finalSplines[i], cam_name, laser_name, temp_output, 1);
            output->output_pts_set[i] = temp_output->output_pts;
        }
        output->spline_set[i] = finalSplines[i];
    }
}