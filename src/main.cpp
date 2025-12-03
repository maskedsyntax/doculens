#include "document_scanner.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  // Use a normal landscape frame size
  FrameSize frameSize{640, 480}; // width, height

  // Video path (can be overridden by argv[1])
  std::string url;
  if (argc > 1) {
    url = argv[1];
  } else {
    url = "/home/batman/maskedsyntax/doculens/testvideo.mp4";
  }

  cv::VideoCapture cap(url);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video: " << url << std::endl;
    return 1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, frameSize.width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameSize.height);
  cap.set(cv::CAP_PROP_BRIGHTNESS, 150);

  cv::Mat img, imgThres, imgContour, imgWarped;

  // Try to respect FPS so video doesn't look sped up
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0)
    fps = 30.0;
  int delay = static_cast<int>(1000.0 / fps);

  while (true) {
    bool success = cap.read(img);
    if (!success || img.empty()) {
      std::cout << "End of video or cannot read frame" << std::endl;
      break;
    }

    cv::resize(img, img, cv::Size(frameSize.width, frameSize.height));
    imgContour = img.clone();

    imgThres = preProcessing(img);
    std::vector<cv::Point> biggest = getContours(imgThres, imgContour);

    bool hasWarp = false;
    if (!biggest.empty()) {
      imgWarped = getWarp(img, biggest, frameSize);
      hasWarp = !imgWarped.empty();
    }

    std::vector<std::vector<cv::Mat>> imageArray;
    if (hasWarp) {
      imageArray = {{img, imgThres}, {imgContour, imgWarped}};
    } else {
      imageArray = {{img, imgThres}, {imgContour, img}};
    }

    cv::Mat stackedImages = stackImages(0.6f, imageArray);

    cv::imshow("Work Flow", stackedImages);
    if (hasWarp) {
      cv::imshow("Result", imgWarped);
    } else {
      cv::imshow("Result", img);
    }

    char key = static_cast<char>(cv::waitKey(delay));
    if (key == 'q' || key == 27) { // q or ESC
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
