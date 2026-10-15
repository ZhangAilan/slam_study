#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#ifndef DEFAULT_IMAGE_1
#define DEFAULT_IMAGE_1 "../2Dto2D_Epipolar_Geometry/1.png"
#endif

#ifndef DEFAULT_IMAGE_2
#define DEFAULT_IMAGE_2 "../2Dto2D_Epipolar_Geometry/2.png"
#endif

#ifndef DEFAULT_DEPTH_IMAGE
#define DEFAULT_DEPTH_IMAGE "../2Dto2D_Epipolar_Geometry/1_depth.png"
#endif

namespace {

const double kRatioThreshold = 0.75;
const double kRansacProbability = 0.999;
const double kRansacThresholdPixels = 1.0;

std::vector<cv::DMatch> ratioTest(
    const std::vector<std::vector<cv::DMatch>>& knn_matches) {
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
    if (argc != 1 && argc != 4) {
        std::cerr << "用法: " << argv[0]
                  << " [图像1 图像2 图像1深度图]\n";
        return 1;
    }

    const std::string image1_path = argc == 4 ? argv[1] : DEFAULT_IMAGE_1;
    const std::string image2_path = argc == 4 ? argv[2] : DEFAULT_IMAGE_2;
    const std::string depth_path = argc == 4 ? argv[3] : DEFAULT_DEPTH_IMAGE;
    const cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat depth_image =
        cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (image1.empty() || image2.empty()) {
        std::cerr << "错误：无法读取输入图像。\n"
                  << "  图像1: " << image1_path << '\n'
                  << "  图像2: " << image2_path << '\n';
        return 1;
    }
    if (depth_image.empty()) {
        std::cerr << "错误：无法读取图像1深度图: " << depth_path << '\n';
        return 1;
    }
    if (depth_image.size() != image1.size()) {
        std::cerr << "错误：图像1深度图尺寸必须与图像1一致。\n"
                  << "  图像1: " << image1.cols << 'x' << image1.rows << '\n'
                  << "  深度图: " << depth_image.cols << 'x' << depth_image.rows
                  << '\n';
        return 1;
    }
    if (depth_image.type() != CV_16UC1 &&
        depth_image.type() != CV_32FC1) {
        std::cerr << "错误：深度图必须是单通道 16 位无符号整数或 "
                     "32 位浮点图像。\n";
        return 1;
    }

    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 520.9, 0.0, 325.1, 0.0, 521.0, 249.7,
         0.0, 0.0, 1.0);
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    if (descriptors1.empty() || descriptors2.empty() ||
        descriptors1.rows < 2 || descriptors2.rows < 2) {
        std::cerr << "错误：ORB 特征不足，无法进行特征匹配。\n";
        return 1;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
    const std::vector<cv::DMatch> good_matches = ratioTest(knn_matches);
    if (good_matches.size() < 5) {
        std::cerr << "错误：比率测试后仅有 " << good_matches.size()
                  << " 个匹配点；本质矩阵估计至少需要 5 个。\n";
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

    cv::Mat ransac_mask;
    const cv::Mat essential_matrix = cv::findEssentialMat(
        points1, points2, camera_matrix, cv::RANSAC, kRansacProbability,
        kRansacThresholdPixels, 1000, ransac_mask);
    if (essential_matrix.empty()) {
        std::cerr << "错误：本质矩阵估计失败。\n";
        return 1;
    }

    cv::Mat relative_rotation;
    cv::Mat relative_translation;
    const int pose_inlier_count = cv::recoverPose(
        essential_matrix, points1, points2, camera_matrix, relative_rotation,
        relative_translation, ransac_mask);
    if (pose_inlier_count == 0) {
        std::cerr << "错误：相机运动恢复失败，没有匹配点通过正深度检验。\n";
        return 1;
    }

    const cv::Mat flat_pose_mask = ransac_mask.reshape(1, 1);
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(static_cast<std::size_t>(pose_inlier_count));
    image_points.reserve(static_cast<std::size_t>(pose_inlier_count));
    for (std::size_t i = 0; i < good_matches.size(); ++i) {
        if (flat_pose_mask.at<unsigned char>(0, static_cast<int>(i)) == 0) {
            continue;
        }

        const cv::Point2f& point = points1[i];
        const int pixel_x = cvRound(point.x);
        const int pixel_y = cvRound(point.y);
        if (pixel_x < 0 || pixel_x >= depth_image.cols || pixel_y < 0 ||
            pixel_y >= depth_image.rows) {
            continue;
        }

        const double depth =
            depth_image.type() == CV_16UC1
                ? static_cast<double>(
                      depth_image.at<unsigned short>(pixel_y, pixel_x))
                : static_cast<double>(depth_image.at<float>(pixel_y, pixel_x));
        if (!std::isfinite(depth) || depth <= 0.0) {
            continue;
        }

        const double x =
            (point.x - camera_matrix.at<double>(0, 2)) * depth /
            camera_matrix.at<double>(0, 0);
        const double y =
            (point.y - camera_matrix.at<double>(1, 2)) * depth /
            camera_matrix.at<double>(1, 1);
        object_points.emplace_back(static_cast<float>(x),
                                   static_cast<float>(y),
                                   static_cast<float>(depth));
        image_points.push_back(points2[i]);
    }
    if (object_points.size() < 4) {
        std::cerr << "错误：深度图为 " << object_points.size()
                  << " 个位姿内点提供了有效深度；EPnP 至少需要 4 个有效"
                     " 3D-2D 对应点。\n";
        return 1;
    }

    cv::Mat rotation_vector;
    cv::Mat translation;
    if (!cv::solvePnP(object_points, image_points, camera_matrix, cv::noArray(),
                      rotation_vector, translation, false,
                      cv::SOLVEPNP_EPNP)) {
        std::cerr << "错误：OpenCV EPnP 求解失败。\n";
        return 1;
    }

    cv::Mat pnp_rotation;
    cv::Rodrigues(rotation_vector, pnp_rotation);
    const cv::Mat rotation_difference =
        pnp_rotation * relative_rotation.t();
    const double rotation_cosine = std::max(
        -1.0, std::min(1.0,
                       (cv::trace(rotation_difference)[0] - 1.0) * 0.5));
    const double rotation_difference_degrees =
        std::acos(rotation_cosine) * 180.0 / CV_PI;

    const double translation_norm = cv::norm(translation);
    if (!std::isfinite(translation_norm) || translation_norm <= 1e-12) {
        std::cerr << "错误：EPnP 得到无效平移向量，无法比较平移方向。\n";
        return 1;
    }
    const cv::Mat normalized_translation = translation / translation_norm;
    const double translation_cosine = std::max(
        -1.0, std::min(1.0, normalized_translation.dot(relative_translation)));
    const double translation_direction_difference_degrees =
        std::acos(translation_cosine) * 180.0 / CV_PI;

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rotation_vector, translation,
                      camera_matrix, cv::noArray(), projected_points);
    double squared_reprojection_error = 0.0;
    for (std::size_t i = 0; i < projected_points.size(); ++i) {
        const cv::Point2f residual = projected_points[i] - image_points[i];
        squared_reprojection_error += residual.dot(residual);
    }
    const double reprojection_rmse =
        std::sqrt(squared_reprojection_error / projected_points.size());

    std::cout << "相机内参矩阵 K：\n"
              << camera_matrix << "\n\n"
              << "ORB 特征点数量：图像1 = " << keypoints1.size()
              << "，图像2 = " << keypoints2.size() << '\n'
              << "比率测试后的匹配点数量： " << good_matches.size() << '\n'
              << "RANSAC 内点数量： " << cv::countNonZero(ransac_mask) << '/'
              << good_matches.size() << '\n'
              << "正深度检验后的内点数量： " << pose_inlier_count << '/'
              << good_matches.size() << '\n'
              << "有效深度 3D-2D 对应点数量： " << object_points.size()
              << '/' << pose_inlier_count << "\n\n"
              << "recoverPose 相对旋转矩阵 R：\n"
              << relative_rotation << "\n\n"
              << "recoverPose 相对平移方向 t（单位向量）：\n"
              << relative_translation << "\n\n"
              << "EPnP 相对旋转矩阵 R：\n"
              << pnp_rotation << "\n\n"
              << "EPnP 平移向量 t（单位与深度图数值一致）：\n"
              << translation << "\n\n"
              << "与 recoverPose 的旋转角差： "
              << rotation_difference_degrees << " 度\n"
              << "与 recoverPose 的平移方向夹角： "
              << translation_direction_difference_degrees << " 度\n"
              << "EPnP 重投影 RMSE： " << reprojection_rmse << " 像素\n"
              << "说明：平移长度的实际单位取决于深度图的编码比例。\n";
    return 0;
}