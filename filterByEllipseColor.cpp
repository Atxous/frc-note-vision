#include <opencv2/opencv.hpp>
#include <vector>
#include <thresholding.h>
#include <math.h>

using namespace cv;

const Mat camMatrix = (Mat_<double>(3, 3) << 1244.0844, 0, 689.01292,
                                                0, 1273.03374, 464.31946,
                                                0, 0, 1);
const Mat distCoeffs = (Mat_<double>(1, 5) << 0.015186, -0.251248, 0.012155, 0.007086, 0.0);
const double scalingFactor = 0.5;
const Scalar upperRange = Scalar(20, 255, 255);
const Scalar lowerRange = Scalar(3, 80, 100);
const double ellipticalThreshold = 0.9;
const double distanceThreshold = 0.25;
const double pixelInchDepth = 176.5 * 164 * 10;
const double PI = 3.14159265358979323846;

void filterBasedOnEllipticality(const Mat& img, std::vector<std::vector<Point>>& rings) {
    for (std::vector<Point> ring : rings) {
        // Draw ellipse
        if (ring.size() < 5) {
            continue;
        }
        RotatedRect ellipse = fitEllipse(ring);
        double area = contourArea(ring);
        double perimeter = arcLength(ring, true);
        int height = img.rows;
        int width = img.cols;

        Mat contour_img = Mat::zeros(Size(height, width), CV_8UC1);
        drawContours(contour_img, std::vector<std::vector<Point>>{ring}, -1, Scalar(255), FILLED);

        Mat ellipse_img = Mat::zeros(Size(height, width), CV_8UC1);
        cv::ellipse(ellipse_img, ellipse, Scalar(255), FILLED);

        Mat intersection, union_result;
        bitwise_and(contour_img, ellipse_img, intersection);
        bitwise_or(contour_img, ellipse_img, union_result);

        double area_intersection = countNonZero(intersection);
        double area_union = countNonZero(union_result);

        if (area_intersection / area_union < ellipticalThreshold) {
            rings.erase(std::remove(rings.begin(), rings.end(), ring), rings.end());
        }
    }
}

int main() {
    VideoCapture cap(0);
    while (cap.isOpened()){
        Mat img;
        cap.read(img);
        if (img.empty()) {
            continue;
        }
        img = imread("test_notes/WIN_20240312_16_40_09_Pro.jpg");
        // Get new camera matrix after undistortion
        Mat img_shifted;
        Mat undistorted_image;
        Mat newcameramatrix, _;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        newcameramatrix = getOptimalNewCameraMatrix(camMatrix, distCoeffs, Size(1280, 720), 1, Size(1280, 720));
        
        // Apply some filters before analysis
        undistort(img, undistorted_image, camMatrix, distCoeffs, newcameramatrix);
        resize(img, img, Size(), scalingFactor, scalingFactor);
        pyrMeanShiftFiltering(img, img_shifted, 21, 51);

        // We first filter based on color
        Mat edges = filterBasedOnColor(img_shifted, lowerRange, upperRange, kernel);

        // if edges is all black
        if (countNonZero(edges) > 0) {
            // Watershed algorithm
            // First we need to find the sure background
            morphologyEx(edges, edges, MORPH_OPEN, kernel, Point(-1,-1), 2);
            Mat sure_bg;
            dilate(edges, sure_bg, kernel, Point(-1,-1), 3);

            // Apply distance transform
            Mat dist_transform;
            distanceTransform(edges, dist_transform, DIST_L2, 5);
            double max_val;
            minMaxLoc(dist_transform, NULL, &max_val);
            dist_transform.convertTo(dist_transform, CV_8U);
            threshold(dist_transform, dist_transform, distanceThreshold * max_val, 255, THRESH_BINARY | THRESH_OTSU);

            // Find sure foreground
            Mat sure_fg = dist_transform > 0;
            sure_fg.convertTo(sure_fg, CV_8U);

            // Find unknown region (we don't know if it's foreground or background)
            Mat unknown = sure_bg - sure_fg;

            // Apply watershed algorithm
            Mat markers;
            connectedComponents(sure_fg, markers, 8, CV_32S);
            markers = markers + 1;
            markers.setTo(0, unknown == 255);
            watershed(img_shifted, markers);

            imshow("sure_fg", sure_fg);
            imshow("sure_bg", sure_bg);
            imshow("unknown", unknown);
            waitKey(0);

            // Find unique markers
            std::vector<int> marker_vector;
            marker_vector.assign((int*)markers.datastart, (int*)markers.dataend);
            std::sort(marker_vector.begin(), marker_vector.end());
            marker_vector.erase(std::unique(marker_vector.begin(), marker_vector.end()), marker_vector.end());

            // Find contours of each marker
            std::vector<std::vector<Point>> rings;
            for (size_t i = 2; i < marker_vector.size(); ++i) {
                int label = marker_vector[i];
                Mat target = (markers == label) / 255;
                target.convertTo(target, CV_8U);

                std::vector<std::vector<Point>> contours;
                findContours(target, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                if (!contours.empty()) {
                    rings.push_back(contours[0]);
                }
            }

            // Filter based on ellipticality and draw bounding box
            filterBasedOnEllipticality(img, rings);
            for (std::vector<Point> ring : rings) {
                // Draw bounding box
                Rect bbox = boundingRect(ring);
                Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
                circle(img, center, 5, Scalar(255, 0, 0), -1);

                center.x = static_cast<int>(center.x / scalingFactor);
                center.y = static_cast<int>(center.y / scalingFactor);

                // Map the center onto the undistorted image
                std::vector<Point2f> srcPoints = {center};
                std::vector<Point2f> dstPoints;
                undistortPoints(srcPoints, dstPoints, camMatrix, distCoeffs, noArray(), newcameramatrix);
                center = dstPoints[0];
                circle(undistorted_image, center, 5, Scalar(255, 0, 0), -1);

                // also add z coordinate
                double z = pixelInchDepth / (1/scalingFactor * bbox.width);

                // mm per pixel
                double scaler = (1280 / bbox.width * 360) / 1280;
                
                // undistort the center
                Point img_center = Point(640, 480);
                std::vector<Point2f> srcPoints2 = {img_center};
                std::vector<Point2f> dstPoints2;
                img_center = dstPoints2[0];
                undistortPoints(srcPoints2, dstPoints2, camMatrix, distCoeffs, noArray(), newcameramatrix);
                circle(undistorted_image, dstPoints2[0], 5, Scalar(255, 0, 0), -1);

                // Calculate the angle wrt to the center
                double angle = asin((center.x - dstPoints2[0].x) * scaler / z) * 180 / PI;

                // Coords are in (undistorted) image space (x, y, z, angle offset from undistorted center)
                double coords[4] = {static_cast<double>(img_center.x), static_cast<double>(img_center.y), z, angle};
                // Draw a small point at the center
                rectangle(img, bbox, Scalar(0, 255, 0), 2);
            }
        }
        imshow("img", img);
        imshow("undistorted", undistorted_image);
        waitKey(0);
    }
    return 0;
}
