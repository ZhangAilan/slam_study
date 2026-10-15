#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef DEFAULT_IMAGE_1
#define DEFAULT_IMAGE_1 "1.png"
#endif

#ifndef DEFAULT_IMAGE_2
#define DEFAULT_IMAGE_2 "2.png"
#endif

#ifndef DEFAULT_MATCHES_IMAGE
#define DEFAULT_MATCHES_IMAGE "inlier_matches.png"
#endif

#ifndef DEFAULT_DEPTH_IMAGE_1
#define DEFAULT_DEPTH_IMAGE_1 "depth_image_1.png"
#endif

#ifndef DEFAULT_DEPTH_IMAGE_2
#define DEFAULT_DEPTH_IMAGE_2 "depth_image_2.png"
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

cv::Mat projectionMatrix(const cv::Mat& camera_matrix, const cv::Mat& rotation,
                         const cv::Mat& translation) {
    cv::Mat extrinsic = cv::Mat::zeros(3, 4, CV_64F);
    rotation.copyTo(extrinsic(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(extrinsic(cv::Rect(3, 0, 1, 3)));
    return camera_matrix * extrinsic;
}

cv::Mat createDepthColorMap() {
    cv::Mat gradient(1, 256, CV_8UC1);
    for (int i = 0; i < gradient.cols; ++i) {
        gradient.at<unsigned char>(0, i) = static_cast<unsigned char>(i);
    }
    cv::Mat color_map;
    cv::applyColorMap(gradient, color_map, cv::COLORMAP_TURBO);
    return color_map;
}

cv::Scalar depthColor(double depth, double min_depth, double max_depth,
                      const cv::Mat& color_map) {
    const double depth_range = max_depth - min_depth;
    const double normalized_depth =
        depth_range > 0.0 ? (depth - min_depth) / depth_range : 0.5;
    const int color_index = cv::saturate_cast<int>(
        255.0 * std::max(0.0, std::min(1.0, normalized_depth)));
    const cv::Vec3b color = color_map.at<cv::Vec3b>(0, color_index);
    return cv::Scalar(color[0], color[1], color[2]);
}

std::string formatDepth(double depth) {
    std::ostringstream label;
    label << std::fixed << std::setprecision(2) << depth;
    return label.str();
}

cv::Mat drawDepthVisualization(
    const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::DMatch>& matches,
    const std::vector<double>& depths,
    const std::vector<unsigned char>& valid_depths, double min_depth,
    double max_depth, const cv::Mat& color_map, const std::string& title,
    int valid_depth_count, bool use_train_indices) {
    const int legend_height = 108;
    cv::Mat visualization(image.rows + legend_height, image.cols, CV_8UC3,
                          cv::Scalar(28, 31, 36));
    cv::Mat image_view =
        visualization(cv::Rect(0, 0, image.cols, image.rows));
    cv::cvtColor(image, image_view, cv::COLOR_GRAY2BGR);

    for (std::size_t i = 0; i < matches.size(); ++i) {
        if (!valid_depths[i]) {
            continue;
        }
        const int keypoint_index =
            use_train_indices ? matches[i].trainIdx : matches[i].queryIdx;
        const cv::Point2f& point = keypoints[keypoint_index].pt;
        const cv::Scalar color =
            depthColor(depths[i], min_depth, max_depth, color_map);
        cv::circle(image_view, point, 5, cv::Scalar(20, 20, 20), cv::FILLED,
                   cv::LINE_AA);
        cv::circle(image_view, point, 3, color, cv::FILLED, cv::LINE_AA);
    }

    const int legend_top = image.rows;
    cv::putText(visualization, title, cv::Point(16, legend_top + 23),
                cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(245, 245, 245), 1,
                cv::LINE_AA);
    std::ostringstream subtitle;
    subtitle << "Relative scale (not meters) | valid points: "
             << valid_depth_count;
    cv::putText(visualization, subtitle.str(),
                cv::Point(16, legend_top + 43), cv::FONT_HERSHEY_SIMPLEX, 0.43,
                cv::Scalar(190, 195, 202), 1, cv::LINE_AA);

    const int bar_x = 16;
    const int bar_y = legend_top + 54;
    const int bar_width = std::max(1, image.cols - 2 * bar_x);
    const int bar_height = 15;
    cv::Mat gradient(1, 256, CV_8UC1);
    for (int i = 0; i < gradient.cols; ++i) {
        gradient.at<unsigned char>(0, i) = static_cast<unsigned char>(i);
    }
    cv::Mat color_bar;
    cv::applyColorMap(gradient, color_bar, cv::COLORMAP_TURBO);
    cv::resize(color_bar,
               visualization(cv::Rect(bar_x, bar_y, bar_width, bar_height)),
               cv::Size(bar_width, bar_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::rectangle(visualization,
                  cv::Rect(bar_x, bar_y, bar_width, bar_height),
                  cv::Scalar(220, 220, 220), 1);

    const int tick_y1 = bar_y - 2;
    const int tick_y2 = bar_y + bar_height + 2;
    const int middle_x = bar_x + bar_width / 2;
    const int right_x = bar_x + bar_width - 1;
    for (int x : {bar_x, middle_x, right_x}) {
        cv::line(visualization, cv::Point(x, tick_y1), cv::Point(x, tick_y2),
                 cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    }

    const int label_y = legend_top + 91;
    const std::string min_label = "Near " + formatDepth(min_depth);
    const std::string mid_label =
        formatDepth((min_depth + max_depth) * 0.5);
    const std::string max_label = "Far " + formatDepth(max_depth);
    cv::putText(visualization, min_label, cv::Point(bar_x, label_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 228, 232), 1,
                cv::LINE_AA);
    int baseline = 0;
    const cv::Size middle_size =
        cv::getTextSize(mid_label, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
    cv::putText(visualization, mid_label,
                cv::Point(middle_x - middle_size.width / 2, label_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 228, 232), 1,
                cv::LINE_AA);
    const cv::Size max_size =
        cv::getTextSize(max_label, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
    cv::putText(visualization, max_label,
                cv::Point(std::max(bar_x, right_x - max_size.width), label_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 228, 232), 1,
                cv::LINE_AA);
    return visualization;
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

    const auto triangulation_start = std::chrono::steady_clock::now();
    const cv::Mat identity_rotation = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat zero_translation = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat projection1 =
        projectionMatrix(camera_matrix, identity_rotation, zero_translation);
    const cv::Mat projection2 =
        projectionMatrix(camera_matrix, rotation, translation);
    std::vector<cv::Point2f> triangulation_points1;
    std::vector<cv::Point2f> triangulation_points2;
    std::vector<std::size_t> triangulation_indices;
    triangulation_points1.reserve(pose_inlier_matches.size());
    triangulation_points2.reserve(pose_inlier_matches.size());
    triangulation_indices.reserve(pose_inlier_matches.size());
    for (std::size_t i = 0; i < good_matches.size(); ++i) {
        if (flat_pose_mask.at<unsigned char>(0, static_cast<int>(i)) == 0) {
            continue;
        }
        triangulation_points1.push_back(points1[i]);
        triangulation_points2.push_back(points2[i]);
        triangulation_indices.push_back(i);
    }

    cv::Mat homogeneous_points;
    cv::triangulatePoints(projection1, projection2, triangulation_points1,
                          triangulation_points2, homogeneous_points);
    homogeneous_points.convertTo(homogeneous_points, CV_64F);
    std::vector<double> depths1(good_matches.size(), 0.0);
    std::vector<double> depths2(good_matches.size(), 0.0);
    std::vector<unsigned char> valid_depths(good_matches.size(), 0);
    std::vector<double> valid_depth_values;
    valid_depth_values.reserve(triangulation_indices.size() * 2);
    for (std::size_t column = 0; column < triangulation_indices.size(); ++column) {
        const double w = homogeneous_points.at<double>(3, static_cast<int>(column));
        if (!std::isfinite(w) || std::abs(w) < 1e-12) {
            continue;
        }

        const cv::Mat point1 =
            homogeneous_points.col(static_cast<int>(column)).rowRange(0, 3) / w;
        const cv::Mat point2 = rotation * point1 + translation;
        const double depth1 = point1.at<double>(2, 0);
        const double depth2 = point2.at<double>(2, 0);
        if (!std::isfinite(depth1) || !std::isfinite(depth2) || depth1 <= 0.0 ||
            depth2 <= 0.0) {
            continue;
        }

        const std::size_t match_index = triangulation_indices[column];
        depths1[match_index] = depth1;
        depths2[match_index] = depth2;
        valid_depths[match_index] = 1;
        valid_depth_values.push_back(depth1);
        valid_depth_values.push_back(depth2);
    }

    if (valid_depth_values.empty()) {
        std::cerr << "错误：三角化未得到有效的正深度点。\n";
        return 1;
    }
    const std::pair<std::vector<double>::const_iterator,
                    std::vector<double>::const_iterator>
        depth_range = std::minmax_element(valid_depth_values.begin(),
                                          valid_depth_values.end());
    const double min_depth = *depth_range.first;
    const double max_depth = *depth_range.second;

    const cv::Mat depth_color_map = createDepthColorMap();
    int valid_depth_count = 0;
    for (std::size_t i = 0; i < good_matches.size(); ++i) {
        if (!valid_depths[i]) {
            continue;
        }
        ++valid_depth_count;
    }
    const cv::Mat depth_visualization1 = drawDepthVisualization(
        image1, keypoints1, good_matches, depths1, valid_depths, min_depth,
        max_depth, depth_color_map, "Depth in camera 1 (Z1)", valid_depth_count,
        false);
    const cv::Mat depth_visualization2 = drawDepthVisualization(
        image2, keypoints2, good_matches, depths2, valid_depths, min_depth,
        max_depth, depth_color_map, "Depth in camera 2 (Z2)", valid_depth_count,
        true);
    if (!cv::imwrite(DEFAULT_DEPTH_IMAGE_1, depth_visualization1)) {
        std::cerr << "错误：无法将图像1深度可视化结果保存到 "
                  << DEFAULT_DEPTH_IMAGE_1 << '\n';
        return 1;
    }
    if (!cv::imwrite(DEFAULT_DEPTH_IMAGE_2, depth_visualization2)) {
        std::cerr << "错误：无法将图像2深度可视化结果保存到 "
                  << DEFAULT_DEPTH_IMAGE_2 << '\n';
        return 1;
    }
    const auto triangulation_end = std::chrono::steady_clock::now();

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
    const double triangulation_ms =
        std::chrono::duration<double, std::milli>(triangulation_end -
                                                  triangulation_start)
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
              << " 毫秒，对极几何估计 = " << estimation_ms
              << " 毫秒，三角测量与绘图 = " << triangulation_ms << " 毫秒\n"
              << "有效三角化点数量： " << valid_depth_count << '/'
              << pose_inlier_count << '\n'
              << "绘图颜色对应深度范围（两相机坐标系合并）： " << min_depth
              << " ~ " << max_depth << "（相对尺度，非实际米制距离）\n"
              << "内点匹配可视化结果已保存到： " << matches_output << '\n'
              << "图像1深度可视化结果已保存到： " << DEFAULT_DEPTH_IMAGE_1
              << '\n'
              << "图像2深度可视化结果已保存到： " << DEFAULT_DEPTH_IMAGE_2
              << '\n';
    return 0;
}