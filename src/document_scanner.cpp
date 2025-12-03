#include "document_scanner.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

cv::Mat preProcessing(const cv::Mat &img) {
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat blur;
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1);

  // Canny edge detection
  cv::Mat canny;
  cv::Canny(blur, canny, 50, 150);

  // Close small gaps in edges without overgrowing too much
  cv::Mat morph;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(canny, morph, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

  return morph;
}

std::vector<cv::Point> getContours(const cv::Mat &img, cv::Mat &imgContour) {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::Mat imgCopy = img.clone();
  cv::findContours(imgCopy, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Point> biggest;
  double maxArea = 0.0;
  double imgArea = static_cast<double>(img.cols * img.rows);

  // Sort contours by area (largest first)
  std::sort(
      contours.begin(), contours.end(),
      [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
        return cv::contourArea(a) > cv::contourArea(b);
      });

  // Draw all reasonably large contours lightly (debug)
  for (const auto &cnt : contours) {
    double area = cv::contourArea(cnt);
    if (area > 2000.0) {
      cv::drawContours(imgContour, std::vector<std::vector<cv::Point>>{cnt}, -1,
                       cv::Scalar(255, 0, 0), 1);
    }
  }

  for (const auto &cnt : contours) {
    double area = cv::contourArea(cnt);
    if (area < 5000.0)
      continue; // too small

    // Ignore contours that are basically the border of the image
    if (area > 0.95 * imgArea)
      continue;

    double peri = cv::arcLength(cnt, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cnt, approx, 0.02 * peri, true);

    if (approx.size() != 4)
      continue;
    if (!cv::isContourConvex(approx))
      continue;

    // Bounding rectangle – reject if it touches the border
    cv::Rect bound = cv::boundingRect(approx);
    int borderMargin = 8;
    if (bound.x <= borderMargin || bound.y <= borderMargin ||
        bound.x + bound.width >= img.cols - borderMargin ||
        bound.y + bound.height >= img.rows - borderMargin) {
      continue;
    }

    // Rough side length and aspect ratio checks
    double w1 = cv::norm(approx[0] - approx[1]);
    double w2 = cv::norm(approx[2] - approx[3]);
    double h1 = cv::norm(approx[1] - approx[2]);
    double h2 = cv::norm(approx[3] - approx[0]);

    double w = (w1 + w2) / 2.0;
    double h = (h1 + h2) / 2.0;
    if (w < 50 || h < 50)
      continue;

    double ratio = w / h;
    if (ratio < 0.4 || ratio > 3.0)
      continue;

    if (area > maxArea) {
      maxArea = area;
      biggest = approx;
    }
  }

  if (!biggest.empty()) {
    // Highlight the chosen document contour in green and thicker
    cv::drawContours(imgContour, std::vector<std::vector<cv::Point>>{biggest},
                     -1, cv::Scalar(0, 255, 0), 4);
  }

  return biggest;
}

std::vector<cv::Point2f> reorder(const std::vector<cv::Point> &points) {
  std::vector<cv::Point2f> pts(4);
  for (int i = 0; i < 4; ++i) {
    pts[i] = cv::Point2f(static_cast<float>(points[i].x),
                         static_cast<float>(points[i].y));
  }

  std::vector<cv::Point2f> newPts(4);

  // x + y
  std::vector<float> sums(4);
  for (int i = 0; i < 4; ++i)
    sums[i] = pts[i].x + pts[i].y;

  int minSumIdx = std::min_element(sums.begin(), sums.end()) - sums.begin();
  int maxSumIdx = std::max_element(sums.begin(), sums.end()) - sums.begin();

  newPts[0] = pts[minSumIdx]; // top-left
  newPts[3] = pts[maxSumIdx]; // bottom-right

  // x - y
  std::vector<float> diffs(4);
  for (int i = 0; i < 4; ++i)
    diffs[i] = pts[i].x - pts[i].y;

  int minDiffIdx = std::min_element(diffs.begin(), diffs.end()) - diffs.begin();
  int maxDiffIdx = std::max_element(diffs.begin(), diffs.end()) - diffs.begin();

  newPts[1] = pts[minDiffIdx]; // top-right
  newPts[2] = pts[maxDiffIdx]; // bottom-left

  return newPts;
}

cv::Mat getWarp(const cv::Mat &img, const std::vector<cv::Point> &biggest,
                const FrameSize &frameSize) {
  cv::Mat imgWarp;

  std::vector<cv::Point2f> pts1 = reorder(biggest);
  if (pts1.size() != 4) {
    return imgWarp; // empty
  }

  std::vector<cv::Point2f> pts2 = {
      cv::Point2f(0.0f, 0.0f),
      cv::Point2f(static_cast<float>(frameSize.width), 0.0f),
      cv::Point2f(0.0f, static_cast<float>(frameSize.height)),
      cv::Point2f(static_cast<float>(frameSize.width),
                  static_cast<float>(frameSize.height))};

  cv::Mat matrix = cv::getPerspectiveTransform(pts1, pts2);
  cv::warpPerspective(img, imgWarp, matrix,
                      cv::Size(frameSize.width, frameSize.height));

  if (imgWarp.empty()) {
    return imgWarp;
  }

  // Optional: crop 20 pixels from all sides then resize back
  const int cropMargin = 20;
  cv::Rect roi(cropMargin, cropMargin, frameSize.width - 2 * cropMargin,
               frameSize.height - 2 * cropMargin);

  roi &= cv::Rect(0, 0, imgWarp.cols, imgWarp.rows);
  if (roi.width <= 0 || roi.height <= 0) {
    return imgWarp; // fallback to full warp if crop invalid
  }

  cv::Mat imgCropped = imgWarp(roi).clone();
  cv::resize(imgCropped, imgCropped,
             cv::Size(frameSize.width, frameSize.height));

  return imgCropped;
}

// Helper to ensure an image is BGR with desired size
static cv::Mat prepareImage(const cv::Mat &img, const cv::Size &size,
                            float scale) {
  cv::Mat resized, colored;

  if (img.empty()) {
    resized = cv::Mat::zeros(size, CV_8UC3);
    return resized;
  }

  cv::Size scaledSize(static_cast<int>(size.width * scale),
                      static_cast<int>(size.height * scale));

  if (img.size() == size) {
    cv::resize(img, resized, scaledSize);
  } else {
    cv::resize(img, resized, size);
    cv::resize(resized, resized, scaledSize);
  }

  if (resized.channels() == 1) {
    cv::cvtColor(resized, colored, cv::COLOR_GRAY2BGR);
  } else {
    colored = resized;
  }

  return colored;
}

cv::Mat stackImages(float scale,
                    const std::vector<std::vector<cv::Mat>> &imgArray) {
  if (imgArray.empty() || imgArray[0].empty()) {
    return cv::Mat();
  }

  int rows = static_cast<int>(imgArray.size());
  int cols = static_cast<int>(imgArray[0].size());

  cv::Size refSize = imgArray[0][0].size();
  if (refSize.width == 0 || refSize.height == 0) {
    return cv::Mat();
  }

  std::vector<cv::Mat> horImages;
  horImages.reserve(rows);

  for (int r = 0; r < rows; ++r) {
    std::vector<cv::Mat> rowImages;
    rowImages.reserve(cols);

    for (int c = 0; c < cols; ++c) {
      cv::Mat prepared = prepareImage(imgArray[r][c], refSize, scale);
      rowImages.push_back(prepared);
    }

    cv::Mat hor;
    cv::hconcat(rowImages, hor);
    horImages.push_back(hor);
  }

  cv::Mat ver;
  cv::vconcat(horImages, ver);
  return ver;
}
