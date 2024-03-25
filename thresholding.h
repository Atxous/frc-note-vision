#ifndef THRESHOLDING_H
#define THRESHOLDING_H

namespace cv {
    class Mat;
}

cv::Mat filterBasedOnColor(const cv::Mat& img, const cv::Scalar& lowerRange, const cv::Scalar& upperRange, const cv::Mat& kernel);

#endif
