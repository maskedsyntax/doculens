# DocuLens: A Document Scanner
A simple document scanning pipeline implemented in C++ with OpenCV.
The program detects the largest 4-point contour in each frame of a video or webcam feed, applies a perspective transform, and shows a flattened, top-down “scan” of the document.

## Steps
* Preprocessing (grayscale -> blur -> Canny -> dilate -> erode)
* Contour detection and 4-point polygon approximation
* Automatic ordering of detected points
* Perspective warp to get a top-down scan
* 2×2 stacked debug view (original, threshold, contour view, warped result)
* Live processing from video file or webcam

## Requirements
* C++17
* OpenCV 4.x
* CMake >= 3.10

Install OpenCV (on Linux):
```bash
sudo apt install libopencv-dev
```

## Build
Run `build.sh` to build the project.

## Run
```bash
./build/document_scanner ./assets/testvideo.mp4
```

## Output Windows

The program displays two windows:
1. Work Flow (2×2 grid)
* Original frame
* Thresholded frame
* Contours
* Warped (scanned) document

2. Result
* Clean final warp of the detected document

Press `q` to quit.

> [!NOTE]
> No ML, just pure classical OpenCV contour detection. Hence, works best with well-lit videos where the document edge is clear.
