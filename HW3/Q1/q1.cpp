/*
Steps:
cd HW3/Q1
g++ q1.cpp -o q1 `pkg-config --cflags --libs opencv4`
./q1

Q1
a) Downloaded the calibration_images
b) Solve the NLS problem
- For each image i, find locations of the 48 corner points in the images
w/ OpenCV findChessboardCorners function.
- For each of the feature point locations in each of the 8 images, construct + solve
NLS problem w/ calibrateCamera in OpenCV.
*/

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

int main(){

    std::vector<std::string> paths = {
        "calibration_images/IMG_3910.JPEG",
        "calibration_images/IMG_3913.JPEG",
        "calibration_images/IMG_3914.JPEG",
        "calibration_images/IMG_3915.JPEG",
        "calibration_images/IMG_3916.JPEG",
        "calibration_images/IMG_3917.JPEG",
        "calibration_images/IMG_3918.JPEG",
        "calibration_images/IMG_3919.JPEG"
    };

    /*
     * Load each image into CV form.
     */
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> clones;
    for(const auto &path : paths){
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        cv::Mat clone = img.clone();

        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image." << std::endl;
            return -1;
        }

        images.push_back(img);
        clones.push_back(clone);
    }

    std::vector<bool> founds;
    cv::Size boardSize(8, 6);
    std::vector<std::vector<cv::Point2f>> corners(images.size());
    int i = 0;

    /*
     * Find the 48 chessboard corners for each image in the vector list.
     */
    for(const auto &img: images){
        bool found = cv::findChessboardCorners(img, boardSize, corners[i]);

        founds.push_back(found);

        // Print out the corner points.
        // cv::drawChessboardCorners(clones[i], boardSize, corners[i], found);
        // cv::imshow("detection", clones[i]);
        // cv::waitKey(1000);
        i++;
    }


    /*
     * TODO(): Complete the CalibrateCamera() part for each image.
     * If the difference between fx and fy is too great, then rows/cols are switched in boardSize.
     * cv::calibrateCamera()
     */
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    std::vector<cv::Point3f> objp;
    float sz = 100; // 0.01 m or 100 mm
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objp.push_back(cv::Point3f(j * sz, i * sz, 0));
        }
    }

    for (size_t i = 0; i < founds.size(); ++i) {
        if (founds[i]) {
            objectPoints.push_back(objp);
            imagePoints.push_back(corners[i]);
        }
    }

    // cv::OutputArray stdDeviationsInts, stdDeviationExts, perViewErrs;
    std::vector<cv::Mat> rvecs(objectPoints.size());
    std::vector<cv::Mat> tvecs(objectPoints.size());

    cv::Mat cameraMat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    // cv::InputOutputArray cameraMat, distCoeffs;

    /*
     * TODO() Need to calibrate for each image.
     */
    cv::Size imgSize = images[0].size();
    double rms = cv::calibrateCamera(
        objectPoints, 
        imagePoints, 
        imgSize,
        cameraMat,
        distCoeffs, 
        rvecs, 
        tvecs
    );
    std::cout << rms << std::endl;
    std::cout << cameraMat << std::endl;
        
    /*
    [1482.258658843503, 0, 549.5399528638159]
    [0, 1476.486249356242, 936.7846042022142]
    [0, 0, 1]
    */

    return 0;
}