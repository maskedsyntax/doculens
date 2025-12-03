#ifndef DOCUMENT_SCANNER_HPP
#define DOCUMENT_SCANNER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct FrameSize {
  int width;
  int height;
};

cv::Mat preProcessing(const cv::Mat &img);

std::vector<cv::Point> getContours(const cv::Mat &img, cv::Mat &imgContour);

std::vector<cv::Point2f> reorder(const std::vector<cv::Point> &points);

cv::Mat getWarp(const cv::Mat &img, const std::vector<cv::Point> &biggest,
                const FrameSize &frameSize);

// 2D stacking
cv::Mat stackImages(float scale,
                    const std::vector<std::vector<cv::Mat>> &imgArray);

#endif // DOCUMENT_SCANNER_HPP
