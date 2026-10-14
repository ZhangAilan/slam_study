#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#ifndef DEFAULT_IMAGE_1
#define DEFAULT_IMAGE_1 "1.png"
#endif

#ifndef DEFAULT_IMAGE_2
#define DEFAULT_IMAGE_2 "2.png"
#endif

#ifndef DEFAULT_MATCHES_IMAGE
#define DEFAULT_MATCHES_IMAGE "inlier_matches.png"
#endif

namespace {

const double kRatioThreshold = 0.75;
const double kRansacProbability = 0.999;
const double kRansacThresholdPixels = 1.0;

std::vector<cv::DMatch> ratioTest(const std::vector<std::vector<cv::DMatch>>&
                                      knn_matches) {
    std::vector<cv::DMatch> good_matches;
    for (const std::vector<cv::DMatch>& neighbors : knn_matches) {
        if (neighbors.size() >= 2 &&
            neighbors[0].distance < kRatioThreshold * neighbors[1].distance) {
            good_matches.push_back(neighbors[0]);
        }
    }
    return good_matches;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 1 && (argc < 3 || argc > 4)) {
        std::cerr << "用法: " << argv[0]
                  << " [图像1 图像2 [匹配结果图路径]]\n";
        return 1;
    }

    const std::string image1_path = argc >= 3 ? argv[1] : DEFAULT_IMAGE_1;
    const std::string image2_path = argc >= 3 ? argv[2] : DEFAULT_IMAGE_2;
    const std::string matches_output =
        argc == 4 ? argv[3] : DEFAULT_MATCHES_IMAGE;

    const cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "错误：无法读取输入图像。\n"
                  << "  图像1: " << image1_path << '\n'
                  << "  图像2: " << image2_path << '\n';
        return 1;
    }

    const cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;

    const auto extraction_start = std::chrono::steady_clock::now();
    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    const auto extraction_end = std::chrono::steady_clock::now();

    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "错误：ORB 无法从两幅图像中提取特征描述子。\n";
        return 1;
    }
    if (descriptors1.rows < 2 || descriptors2.rows < 2) {
        std::cerr << "错误：比率测试匹配要求每幅图像至少有两个特征描述子。\n";
        return 1;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    const auto matching_start = std::chrono::steady_clock::now();
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
    const std::vector<cv::DMatch> good_matches = ratioTest(knn_matches);
    const auto matching_end = std::chrono::steady_clock::now();

    if (good_matches.size() < 5) {
        std::cerr << "错误：比率测试后仅有 " << good_matches.size()
                  << " 个可靠匹配点；估计本质矩阵至少需要 5 个。\n";
        return 1;
    }

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(good_matches.size());
    points2.reserve(good_matches.size());
    for (const cv::DMatch& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 520.9, 0.0, 325.1, 0.0, 521.0, 249.7, 0.0,
         0.0, 1.0);
    cv::Mat ransac_mask;
    const auto estimation_start = std::chrono::steady_clock::now();
    const cv::Mat essential_matrix = cv::findEssentialMat(
        points1, points2, camera_matrix, cv::RANSAC, kRansacProbability,
        kRansacThresholdPixels, 1000, ransac_mask);
    if (essential_matrix.empty()) {
        std::cerr << "错误：本质矩阵估计失败。\n";
        return 1;
    }

    cv::Mat rotation;
    cv::Mat translation;
    cv::Mat pose_mask = ransac_mask.clone();
    const int pose_inlier_count = cv::recoverPose(
        essential_matrix, points1, points2, camera_matrix, rotation, translation,
        pose_mask);
    const auto estimation_end = std::chrono::steady_clock::now();
    if (pose_inlier_count == 0) {
        std::cerr << "错误：相机运动恢复失败，没有匹配点通过正深度检验。\n";
        return 1;
    }

    const cv::Mat fundamental_matrix =
        camera_matrix.inv().t() * essential_matrix * camera_matrix.inv();
    const cv::Mat flat_pose_mask = pose_mask.reshape(1, 1);
    std::vector<cv::DMatch> pose_inlier_matches;
    pose_inlier_matches.reserve(static_cast<std::size_t>(pose_inlier_count));
    for (std::size_t i = 0; i < good_matches.size(); ++i) {
        if (flat_pose_mask.at<unsigned char>(0, static_cast<int>(i)) != 0) {
            pose_inlier_matches.push_back(good_matches[i]);
        }
    }

    cv::Mat match_visualization;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, pose_inlier_matches,
                    match_visualization);
    if (!cv::imwrite(matches_output, match_visualization)) {
        std::cerr << "错误：无法将匹配可视化结果保存到 "
                  << matches_output << '\n';
        return 1;
    }

    const double extraction_ms =
        std::chrono::duration<double, std::milli>(extraction_end -
                                                  extraction_start)
            .count();
    const double matching_ms =
        std::chrono::duration<double, std::milli>(matching_end - matching_start)
            .count();
    const double estimation_ms =
        std::chrono::duration<double, std::milli>(estimation_end -
                                                  estimation_start)
            .count();

    std::cout << "相机内参矩阵 K：\n"
              << camera_matrix << "\n\n"
              << "ORB 特征点数量：图像1 = " << keypoints1.size()
              << "，图像2 = " << keypoints2.size() << '\n'
              << "比率测试后的可靠匹配点数量（阈值 "
              << kRatioThreshold << "）： " << good_matches.size() << '\n'
              << "RANSAC 内点数量： " << cv::countNonZero(ransac_mask) << '/'
              << good_matches.size() << '\n'
              << "正深度检验后的位姿内点数量： " << pose_inlier_count << '/'
              << good_matches.size() << "\n\n"
              << "本质矩阵 E：\n"
              << essential_matrix << "\n\n"
              << "基础矩阵 F = K^-T * E * K^-1：\n"
              << fundamental_matrix << "\n\n"
              << "相对旋转矩阵 R（从图像1相机坐标系到图像2相机坐标系）：\n"
              << rotation << "\n\n"
              << "相对平移方向 t（单位向量；平移尺度无法由两幅图像确定）：\n"
              << translation << "\n\n"
              << "运行耗时：ORB 特征提取 = " << extraction_ms
              << " 毫秒，特征匹配 = " << matching_ms
              << " 毫秒，对极几何估计 = " << estimation_ms << " 毫秒\n"
              << "内点匹配可视化结果已保存到： " << matches_output << '\n';
    return 0;
}