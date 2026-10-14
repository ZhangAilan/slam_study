#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

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

int main(int argc, char* argv[]) {
    if (argc != 1 && (argc < 3 || argc > 4)) {
        std::cerr << "Usage: " << argv[0]
                  << " [imageA imageB [outputImage]]\n";
        return 1;
    }

    const std::string image_a_path = argc >= 3 ? argv[1] : DEFAULT_IMAGE_A;
    const std::string image_b_path = argc >= 3 ? argv[2] : DEFAULT_IMAGE_B;
    const std::string output_path = argc == 4 ? argv[3] : DEFAULT_OUTPUT_IMAGE;

    const cv::Mat image_a = cv::imread(image_a_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat image_b = cv::imread(image_b_path, cv::IMREAD_GRAYSCALE);
    if (image_a.empty() || image_b.empty()) {
        std::cerr << "Error: could not read both input images.\n"
                  << "  Image A: " << image_a_path << '\n'
                  << "  Image B: " << image_b_path << '\n';
        return 1;
    }

    const cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints_a;
    std::vector<cv::KeyPoint> keypoints_b;
    cv::Mat descriptors_a;
    cv::Mat descriptors_b;

    const auto extraction_start = std::chrono::steady_clock::now();
    orb->detectAndCompute(image_a, cv::noArray(), keypoints_a, descriptors_a);
    orb->detectAndCompute(image_b, cv::noArray(), keypoints_b, descriptors_b);
    const auto extraction_end = std::chrono::steady_clock::now();

    if (descriptors_a.empty() || descriptors_b.empty()) {
        std::cerr << "Error: ORB could not extract descriptors from both images.\n";
        return 1;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    const auto matching_start = std::chrono::steady_clock::now();
    matcher.match(descriptors_a, descriptors_b, matches);
    const auto matching_end = std::chrono::steady_clock::now();

    if (matches.empty()) {
        std::cerr << "Error: no feature matches were found.\n";
        return 1;
    }

    const auto distance_bounds = std::minmax_element(
        matches.begin(), matches.end(),
        [](const cv::DMatch& left, const cv::DMatch& right) {
            return left.distance < right.distance;
        });
    const float match_distance_threshold = 2.0f * distance_bounds.first->distance;
    std::vector<cv::DMatch> good_matches;
    std::copy_if(matches.begin(), matches.end(),
                 std::back_inserter(good_matches),
                 [match_distance_threshold](const cv::DMatch& match) {
                     return match.distance < match_distance_threshold;
                 });

    cv::Mat match_image;
    cv::drawMatches(image_a, keypoints_a, image_b, keypoints_b, good_matches,
                    match_image);
    for (const cv::DMatch& match : good_matches) {
        const cv::Point point_a(
            cvRound(keypoints_a[match.queryIdx].pt.x),
            cvRound(keypoints_a[match.queryIdx].pt.y));
        const cv::Point point_b(
            cvRound(keypoints_b[match.trainIdx].pt.x) + image_a.cols,
            cvRound(keypoints_b[match.trainIdx].pt.y));
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

    std::cout << "Feature extraction time: " << extraction_ms << " ms\n"
              << "Feature matching time: " << matching_ms << " ms\n"
              << "Keypoints in image A: " << keypoints_a.size() << '\n'
              << "Keypoints in image B: " << keypoints_b.size() << '\n'
              << "Matches before filtering: " << matches.size() << '\n'
              << "Filtering threshold (2 * min_dist): "
              << match_distance_threshold << '\n'
              << "Matches after filtering: " << good_matches.size() << '\n'
              << "max_dist: " << distance_bounds.second->distance << '\n'
              << "min_dist: " << distance_bounds.first->distance << '\n'
              << "Match visualization saved to: " << output_path << '\n';
    return 0;
}
