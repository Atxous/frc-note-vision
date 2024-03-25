#include <opencv2/opencv.hpp>
#include <vector>
#include <thresholding.h>

using namespace cv;

Mat filterBasedOnColor(const Mat& img, const Scalar& lowerRange, const Scalar& upperRange, const Mat& kernel) {
    Mat img_blur;
    GaussianBlur(img, img_blur, Size(7, 7), 0);

    Mat edges, orange_mask;
    Canny(img_blur, edges, 100, 255);
    cvtColor(img_blur, img_blur, COLOR_BGR2HSV);
    inRange(img_blur, lowerRange, upperRange, orange_mask);
    
    
    dilate(orange_mask, orange_mask, kernel, Point(-1,-1), 2);
    dilate(edges, edges, kernel, Point(-1,-1), 1);

    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); ++i) {
        if (contourArea(contours[i]) > 5) {
            Rect bounding_rect = boundingRect(contours[i]);
            Mat roi = orange_mask(bounding_rect);
            if (countNonZero(roi) == 0) {
                drawContours(edges, contours, i, Scalar(0, 0, 0), -1);
            }
            else {
                drawContours(edges, contours, i, Scalar(255, 255, 255), -1);
            }
        }
    }
    return edges;
}