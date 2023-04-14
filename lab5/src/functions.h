#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include "aruco_samples_utility.hpp"


enum flags { fix_point, zero_tg_dist, fix_ratio };
void calibrateCamera(flags flag);
void createMarker();
bool readDetectorParameters(std::string filename, cv::aruco::DetectorParameters& params);
void detectMarker();
void drawCubes(cv::Mat& image, const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
