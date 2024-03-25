import cv2
import numpy as np

CAMERA_MATRIX = np.array([[1244.0844, 0, 689.01292],
                          [0, 1273.03374, 464.31946],
                          [0, 0, 1]])
PROJECTION_MATRIX = np.array([[1221.78394, 0, 700.05636, 0],
                              [0, 1265.19751, 468.77513, 0],
                              [0, 0, 1, 0]])
DIST_COEFFS = np.array([0.015186, -0.251248, 0.012155, 0.007086, 0.0])
UPPER_RANGE = np.array([20, 255, 255])
LOWER_RANGE = np.array([3, 80, 100])
SCALING_FACTOR = 0.5
ELLIPTICAL_THRESHOLD = 0.9

newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(
    CAMERA_MATRIX, DIST_COEFFS, (1280, 720), 1, (1280, 720)
)
img = cv2.imread(r"test_set\WIN_20240312_16_40_13_Pro.jpg")
undistorted_image = cv2.undistort(img, CAMERA_MATRIX, DIST_COEFFS, None, newcameramatrix)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

img = cv2.resize(img, (0, 0), fx = SCALING_FACTOR, fy = SCALING_FACTOR)
# apply pyramid mean shift filtering
img_shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
img_blur = cv2.GaussianBlur(img_shifted, (7, 7), 0)

edges = cv2.Canny(image = img_blur, threshold1 = 100, threshold2 = 255) # Canny Edge Detection
orange_mask = cv2.inRange(cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV), LOWER_RANGE, UPPER_RANGE)
orange_mask = cv2.dilate(orange_mask, kernel, iterations = 2)
edges = cv2.dilate(edges, kernel, iterations = 1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# now, we filter out the contours that do not have orange pixels and consider them to be background
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 5:
        x, y, w, h = cv2.boundingRect(contours[i])
        if cv2.countNonZero(orange_mask[y:y+h, x:x+w]) == 0:
            cv2.drawContours(edges, contours, i, (0, 0, 0), -1)
        else:
            cv2.drawContours(edges, contours, i, (255, 255, 255), -1)
            
# threshold (0, 255, 0)
opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 2)
sure_bg = cv2.dilate(edges, kernel, iterations = 3)

dist_transform = np.uint8(cv2.distanceTransform(opening, cv2.DIST_L2, 5))
ret, sure_fg = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown==255] = 0
markers = cv2.watershed(img_shifted, markers)

labels = np.unique(markers)
rings = []
for label in labels[2:]:
    target = np.where(markers == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings.append(contours[0])

for ring in rings:
    # draw ellipse
    ellipse = cv2.fitEllipse(ring)
    area = cv2.contourArea(ring)
    perimeter = cv2.arcLength(ring, True)
    height, width = img.shape[:2]
    
    contour_img = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(contour_img, [ring], -1, (255), thickness=cv2.FILLED)
    ellipse_img = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(ellipse_img, ellipse, (255), thickness=cv2.FILLED)
    intersection = cv2.bitwise_and(contour_img, ellipse_img)
    union = cv2.bitwise_or(contour_img, ellipse_img)
    area_intersection = np.count_nonzero(intersection)
    area_union = np.count_nonzero(union)
    if area_intersection/area_union > ELLIPTICAL_THRESHOLD:
        # draw bounding box
        x, y, w, h = cv2.boundingRect(ring)
        
        # print center
        center = (int(x + w/2), int(y + h/2))
        cv2.circle(img, center, 5, (255, 0, 0), -1)
        
        center = (center[0] * 1/SCALING_FACTOR, center[1] * 1/SCALING_FACTOR)
        
        # map the center onto the undistorted image
        center = cv2.undistortPoints(center, CAMERA_MATRIX, DIST_COEFFS, None, newcameramatrix)
        center = (int(center[0][0][0]), int(center[0][0][1]))
        cv2.circle(undistorted_image, center, 5, (255, 0, 0), -1)
        print(center)
        
        # draw a small point at the center
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.imshow("undistorted", undistorted_image)
cv2.waitKey(0)