#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef DEFAULT_IMAGE_A
#define DEFAULT_IMAGE_A "A.jpg"
#endif

#ifndef DEFAULT_IMAGE_B
#define DEFAULT_IMAGE_B "B.jpg"
#endif

#ifndef DEFAULT_OUTPUT_IMAGE
#define DEFAULT_OUTPUT_IMAGE "orb_matches.jpg"
#endif

namespace {

constexpr int kMaxFeatures = 1500;
constexpr int kPyramidLevels = 8;
constexpr float kScaleFactor = 1.2f;
constexpr int kFastThreshold = 20;
constexpr int kBriefPatchRadius = 15;
constexpr float kRatioThreshold = 0.75f;
constexpr int kBriefBits = 256;
constexpr int kBriefBytes = kBriefBits / 8;

struct BriefPair {
    cv::Point first;
    cv::Point second;
};

struct FeatureSet {
    std::vector<cv::KeyPoint> keypoints;
    std::vector<std::array<std::uint8_t, kBriefBytes>> descriptors;
};

const std::array<cv::Point, 16>& fast_circle() {
    static const std::array<cv::Point, 16> circle = {{
        {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0},  {3, 1},
        {2, 2},  {1, 3},  {0, 3},  {-1, 3}, {-2, 2}, {-3, 1},
        {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3},
    }};
    return circle;
}

const std::array<BriefPair, kBriefBits>& brief_pattern() {
    static const std::array<BriefPair, kBriefBits> pattern = [] {
        std::array<BriefPair, kBriefBits> points{};
        std::mt19937 generator(0x4f5242);
        std::uniform_int_distribution<int> coordinate(-kBriefPatchRadius,
                                                       kBriefPatchRadius);
        for (BriefPair& pair : points) {
            auto sample_point = [&]() {
                cv::Point point;
                do {
                    point = cv::Point(coordinate(generator),
                                      coordinate(generator));
                } while (point.dot(point) >
                         kBriefPatchRadius * kBriefPatchRadius);
                return point;
            };
            pair.first = sample_point();
            pair.second = sample_point();
        }
        return points;
    }();
    return pattern;
}

float harris_response(const cv::Mat& image, int x, int y) {
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;
    for (int dy = -3; dy <= 3; ++dy) {
        const std::uint8_t* row_above = image.ptr<std::uint8_t>(y + dy - 1);
        const std::uint8_t* row = image.ptr<std::uint8_t>(y + dy);
        const std::uint8_t* row_below = image.ptr<std::uint8_t>(y + dy + 1);
        for (int dx = -3; dx <= 3; ++dx) {
            const int px = x + dx;
            const double gx =
                static_cast<double>(row[px + 1]) - row[px - 1];
            const double gy =
                static_cast<double>(row_below[px]) - row_above[px];
            xx += gx * gx;
            yy += gy * gy;
            xy += gx * gy;
        }
    }
    return static_cast<float>(xx * yy - xy * xy -
                              0.04 * (xx + yy) * (xx + yy));
}

bool is_fast_corner(const cv::Mat& image, int x, int y, int threshold) {
    const int center = image.ptr<std::uint8_t>(y)[x];
    int brighter_run = 0;
    int darker_run = 0;
    for (int i = 0; i < 16 + 8; ++i) {
        const cv::Point& offset = fast_circle()[i & 15];
        const int value =
            image.ptr<std::uint8_t>(y + offset.y)[x + offset.x];
        brighter_run = value > center + threshold ? brighter_run + 1 : 0;
        darker_run = value < center - threshold ? darker_run + 1 : 0;
        if (brighter_run >= 9 || darker_run >= 9) {
            return true;
        }
    }
    return false;
}

float intensity_centroid_angle(const cv::Mat& image, int x, int y) {
    int m10 = 0;
    int m01 = 0;
    for (int dy = -kBriefPatchRadius; dy <= kBriefPatchRadius; ++dy) {
        const int extent = static_cast<int>(
            std::sqrt(kBriefPatchRadius * kBriefPatchRadius - dy * dy));
        const std::uint8_t* row = image.ptr<std::uint8_t>(y + dy);
        for (int dx = -extent; dx <= extent; ++dx) {
            const int intensity = row[x + dx];
            m10 += dx * intensity;
            m01 += dy * intensity;
        }
    }
    return static_cast<float>(std::atan2(static_cast<double>(m01),
                                         static_cast<double>(m10)) *
                              180.0 / CV_PI);
}

std::array<std::uint8_t, kBriefBytes> compute_descriptor(
    const cv::Mat& image, int x, int y, float angle_degrees) {
    std::array<std::uint8_t, kBriefBytes> descriptor{};
    const double angle = angle_degrees * CV_PI / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const auto& pattern = brief_pattern();

    for (int bit = 0; bit < kBriefBits; ++bit) {
        const BriefPair& pair = pattern[bit];
        const auto rotate = [cosine, sine](const cv::Point& point) {
            return cv::Point(cvRound(cosine * point.x - sine * point.y),
                             cvRound(sine * point.x + cosine * point.y));
        };
        const cv::Point first = rotate(pair.first);
        const cv::Point second = rotate(pair.second);
        const std::uint8_t first_value =
            image.ptr<std::uint8_t>(y + first.y)[x + first.x];
        const std::uint8_t second_value =
            image.ptr<std::uint8_t>(y + second.y)[x + second.x];
        if (first_value < second_value) {
            descriptor[bit / 8] |=
                static_cast<std::uint8_t>(1U << (bit % 8));
        }
    }
    return descriptor;
}

FeatureSet extract_features(const cv::Mat& image) {
    FeatureSet features;
    std::vector<cv::Mat> pyramid;
    pyramid.reserve(kPyramidLevels);
    for (int level = 0; level < kPyramidLevels; ++level) {
        const double scale = std::pow(kScaleFactor, level);
        cv::Mat level_image;
        cv::resize(image, level_image,
                   cv::Size(cvRound(image.cols / scale),
                            cvRound(image.rows / scale)),
                   0.0, 0.0, cv::INTER_LINEAR);
        pyramid.push_back(std::move(level_image));
    }

    const double feature_scale =
        (1.0 - 1.0 / kScaleFactor) /
        (1.0 - std::pow(1.0 / kScaleFactor, kPyramidLevels));
    int remaining_features = kMaxFeatures;
    int target_features = 0;

    for (int level = 0; level < kPyramidLevels; ++level) {
        if (level == kPyramidLevels - 1) {
            target_features = remaining_features;
        } else {
            target_features =
                cvRound(kMaxFeatures * feature_scale *
                        std::pow(1.0 / kScaleFactor, level));
            remaining_features -= target_features;
        }

        const cv::Mat& level_image = pyramid[level];
        if (level_image.cols <= 2 * (kBriefPatchRadius + 1) ||
            level_image.rows <= 2 * (kBriefPatchRadius + 1)) {
            continue;
        }

        cv::Mat response = cv::Mat::zeros(level_image.size(), CV_32F);
        for (int y = kBriefPatchRadius + 1;
             y < level_image.rows - kBriefPatchRadius - 1; ++y) {
            for (int x = kBriefPatchRadius + 1;
                 x < level_image.cols - kBriefPatchRadius - 1; ++x) {
                if (is_fast_corner(level_image, x, y, kFastThreshold)) {
                    response.at<float>(y, x) = harris_response(level_image, x, y);
                }
            }
        }

        std::vector<cv::KeyPoint> candidates;
        for (int y = kBriefPatchRadius + 2;
             y < level_image.rows - kBriefPatchRadius - 2; ++y) {
            for (int x = kBriefPatchRadius + 2;
                 x < level_image.cols - kBriefPatchRadius - 2; ++x) {
                const float score = response.at<float>(y, x);
                if (score <= 0.0f) {
                    continue;
                }
                bool local_maximum = true;
                for (int dy = -1; dy <= 1 && local_maximum; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if ((dx != 0 || dy != 0) &&
                            response.at<float>(y + dy, x + dx) > score) {
                            local_maximum = false;
                            break;
                        }
                    }
                }
                if (local_maximum) {
                    candidates.emplace_back(
                        cv::Point2f(static_cast<float>(x),
                                    static_cast<float>(y)),
                        31.0f, -1.0f, score, level);
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const cv::KeyPoint& left, const cv::KeyPoint& right) {
                      return left.response > right.response;
                  });
        if (static_cast<int>(candidates.size()) > target_features) {
            candidates.resize(target_features);
        }

        const float scale = std::pow(kScaleFactor, level);
        for (cv::KeyPoint& keypoint : candidates) {
            const int x = cvRound(keypoint.pt.x);
            const int y = cvRound(keypoint.pt.y);
            keypoint.angle = intensity_centroid_angle(level_image, x, y);
            const auto descriptor =
                compute_descriptor(level_image, x, y, keypoint.angle);
            keypoint.pt.x *= scale;
            keypoint.pt.y *= scale;
            keypoint.size *= scale;
            features.keypoints.push_back(keypoint);
            features.descriptors.push_back(descriptor);
        }
    }
    return features;
}

int hamming_distance(const std::array<std::uint8_t, kBriefBytes>& left,
                     const std::array<std::uint8_t, kBriefBytes>& right) {
    int distance = 0;
    for (int i = 0; i < kBriefBytes; ++i) {
        distance += __builtin_popcount(
            static_cast<unsigned int>(left[i] ^ right[i]));
    }
    return distance;
}

std::vector<cv::DMatch> ratio_test_matches(const FeatureSet& first,
                                           const FeatureSet& second) {
    std::vector<cv::DMatch> matches;
    if (second.descriptors.size() < 2) {
        return matches;
    }

    for (std::size_t query = 0; query < first.descriptors.size(); ++query) {
        int best_distance = kBriefBits + 1;
        int second_best_distance = kBriefBits + 1;
        int best_train = -1;
        for (std::size_t train = 0; train < second.descriptors.size(); ++train) {
            const int distance =
                hamming_distance(first.descriptors[query],
                                 second.descriptors[train]);
            if (distance < best_distance) {
                second_best_distance = best_distance;
                best_distance = distance;
                best_train = static_cast<int>(train);
            } else if (distance < second_best_distance) {
                second_best_distance = distance;
            }
        }
        if (best_train >= 0 &&
            best_distance < kRatioThreshold * second_best_distance) {
            matches.emplace_back(static_cast<int>(query), best_train,
                                 static_cast<float>(best_distance));
        }
    }
    return matches;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 1 && (argc < 3 || argc > 4)) {
        std::cerr << "Usage: " << argv[0]
                  << " [imageA imageB [outputImage]]\n";
        return 1;
    }

    const std::string image_a_path = argc >= 3 ? argv[1] : DEFAULT_IMAGE_A;
    const std::string image_b_path = argc >= 3 ? argv[2] : DEFAULT_IMAGE_B;
    const std::string output_path = argc == 4 ? argv[3] : DEFAULT_OUTPUT_IMAGE;

    const cv::Mat color_a = cv::imread(image_a_path, cv::IMREAD_COLOR);
    const cv::Mat color_b = cv::imread(image_b_path, cv::IMREAD_COLOR);
    if (color_a.empty() || color_b.empty()) {
        std::cerr << "Error: could not read both input images.\n"
                  << "  Image A: " << image_a_path << '\n'
                  << "  Image B: " << image_b_path << '\n';
        return 1;
    }
    cv::Mat gray_a;
    cv::Mat gray_b;
    cv::cvtColor(color_a, gray_a, cv::COLOR_BGR2GRAY);
    cv::cvtColor(color_b, gray_b, cv::COLOR_BGR2GRAY);

    const auto extraction_start = std::chrono::steady_clock::now();
    const FeatureSet features_a = extract_features(gray_a);
    const FeatureSet features_b = extract_features(gray_b);
    const auto extraction_end = std::chrono::steady_clock::now();

    if (features_a.descriptors.empty() || features_b.descriptors.empty()) {
        std::cerr << "Error: manual ORB extraction found no descriptors in "
                     "one or both images.\n";
        return 1;
    }

    const auto matching_start = std::chrono::steady_clock::now();
    const std::vector<cv::DMatch> good_matches =
        ratio_test_matches(features_a, features_b);
    const auto matching_end = std::chrono::steady_clock::now();
    if (good_matches.empty()) {
        std::cerr << "Error: no matches passed the Lowe ratio test.\n";
        return 1;
    }

    std::vector<cv::DMatch> inlier_matches;
    cv::Mat homography;
    if (good_matches.size() >= 4) {
        std::vector<cv::Point2f> points_a;
        std::vector<cv::Point2f> points_b;
        points_a.reserve(good_matches.size());
        points_b.reserve(good_matches.size());
        for (const cv::DMatch& match : good_matches) {
            points_a.push_back(features_a.keypoints[match.queryIdx].pt);
            points_b.push_back(features_b.keypoints[match.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        homography = cv::findHomography(points_a, points_b, cv::RANSAC, 3.0,
                                        inlier_mask, 2000, 0.99);
        if (!homography.empty() && !inlier_mask.empty()) {
            for (std::size_t i = 0; i < good_matches.size(); ++i) {
                if (inlier_mask.at<std::uint8_t>(static_cast<int>(i))) {
                    inlier_matches.push_back(good_matches[i]);
                }
            }
        }
    }

    const bool has_inliers = !inlier_matches.empty();
    const std::vector<cv::DMatch>& displayed_matches =
        has_inliers ? inlier_matches : good_matches;
    cv::Mat match_image;
    cv::drawMatches(color_a, features_a.keypoints, color_b,
                    features_b.keypoints, displayed_matches, match_image,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    for (const cv::DMatch& match : displayed_matches) {
        const cv::Point point_a(
            cvRound(features_a.keypoints[match.queryIdx].pt.x),
            cvRound(features_a.keypoints[match.queryIdx].pt.y));
        const cv::Point point_b(
            cvRound(features_b.keypoints[match.trainIdx].pt.x) + color_a.cols,
            cvRound(features_b.keypoints[match.trainIdx].pt.y));
        cv::line(match_image, point_a, point_b, cv::Scalar(0, 0, 255), 4,
                 cv::LINE_AA);
        cv::circle(match_image, point_a, 5, cv::Scalar(0, 0, 255), 3,
                   cv::LINE_AA);
        cv::circle(match_image, point_b, 5, cv::Scalar(0, 0, 255), 3,
                   cv::LINE_AA);
    }
    if (!cv::imwrite(output_path, match_image)) {
        std::cerr << "Error: could not write match visualization to "
                  << output_path << '\n';
        return 1;
    }

    const auto extraction_ms =
        std::chrono::duration<double, std::milli>(extraction_end -
                                                   extraction_start)
            .count();
    const auto matching_ms =
        std::chrono::duration<double, std::milli>(matching_end -
                                                   matching_start)
            .count();
    const auto distance_bounds = std::minmax_element(
        good_matches.begin(), good_matches.end(),
        [](const cv::DMatch& left, const cv::DMatch& right) {
            return left.distance < right.distance;
        });

    std::cout << "Feature extraction time: " << extraction_ms << " ms\n"
              << "Feature matching time: " << matching_ms << " ms\n"
              << "Keypoints in image A: " << features_a.keypoints.size() << '\n'
              << "Keypoints in image B: " << features_b.keypoints.size() << '\n'
              << "Matches passing Lowe ratio test: " << good_matches.size()
              << '\n'
              << "RANSAC inliers: " << inlier_matches.size() << '\n'
              << "max_dist: " << distance_bounds.second->distance << '\n'
              << "min_dist: " << distance_bounds.first->distance << '\n'
              << "Displayed matches: " << displayed_matches.size() << '\n'
              << "Match visualization saved to: " << output_path << '\n';
    if (!homography.empty()) {
        std::cout << "Estimated homography:\n" << homography << '\n';
    } else {
        std::cout << "Homography unavailable; displaying ratio-test matches.\n";
    }
    return 0;
}
