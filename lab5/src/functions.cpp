#include "functions.h"
#include <iostream>


void createMarker() {
    cv::Mat marker_image;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50); //�������� ������� �������
    cv::aruco::drawMarker(dictionary, 23, 200, marker_image, 1); //������� ������
    cv::imwrite("marker.png", marker_image); //���������
}

void calibrateCamera(flags flag) {
    std::string output_file = "calibration2.xml"; //����, � ������� ����� �������� ��������� ����������
    cv::String params_filename = "params.yml"; //���� � ����������� ��������� ��������
    const int marker_size = 23;
    const int margin = 12;
    const int num_markers_x = 7;
    const int num_markers_y = 6;    //Create a board
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    //������� ����� �����
    cv::Ptr<cv::aruco::Board> board = cv::aruco::GridBoard::create(num_markers_x, num_markers_y, float(marker_size), float(margin), dictionary);
    
    const float aspect_ratio = 1.0;
    int callibration_flags = 0;

    if (fix_ratio) {    //��������� ����������� �����
        callibration_flags |= cv::CALIB_FIX_ASPECT_RATIO;
    }
    if (zero_tg_dist) {
        callibration_flags |= cv::CALIB_ZERO_TANGENT_DIST;
    }
    if (fix_point) {
        callibration_flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    }
    cv::Ptr<cv::aruco::DetectorParameters> detector_ptr; 
    cv::aruco::DetectorParameters detector_params; 

    readDetectorParameters(params_filename, detector_params);
    detector_ptr = &detector_params;
    bool refindStrategy = true; //��������� ���������� ��������

    cv::VideoCapture input(1);
    input.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    input.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    const int waitTime = 10;
    if (!input.isOpened()) {
        std::cerr << "Video hasn't been captured" << std::endl;
    }

    std::vector<std::vector<std::vector<cv::Point2f>>> all_corners; //��������� ���� ��������
    std::vector<std::vector<int>> all_ids;      //���������� ������ ��������
    cv::Size image_size = cv::Size(1920, 1080);

    while (input.grab()) {
        cv::Mat image, imageCopy;
        input.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;

        cv::aruco::detectMarkers(image, dictionary, corners, ids, detector_ptr, rejected);

        // ��� ������ �������� ��������
        if (refindStrategy) {
            cv::aruco::refineDetectedMarkers(image, board, corners, ids, rejected);
        }
        // ������� ���������� ��������, ������ �������
        image.copyTo(imageCopy);
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        }
        putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
        cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

        imshow("out", imageCopy);
        char key = (char)cv::waitKey(waitTime);
        if (key == 27) break;
        if (key == 'c' && ids.size() > 0) { //���� ���� ��� ����������, ��������� ��������� �������� �����
            std::cout << "Frame captured" << std::endl;
            all_corners.push_back(corners);
            all_ids.push_back(ids);
            image_size = image.size();
        }
    }
    if (all_ids.size() < 1) {       //���� �� ���� ���������� �������
        std::cerr << "Not enough captures for calibration" << std::endl;
        return;
    }

    cv::Mat cameraMatrix, distCoeffs;   //������� ������ � ������������ ���������
    std::vector<cv::Mat> rvecs, tvecs;
    double reprojection_error;  //������ ����������

    // �������������� ������ � ����������
    std::vector<std::vector<cv::Point2f>> all_corners_concatenated;
    std::vector<int> all_ids_concatenated;
    std::vector<int> marker_counter_per_frame;
    marker_counter_per_frame.reserve(all_corners.size());
    for (unsigned int i = 0; i < all_corners.size(); i++) {
        marker_counter_per_frame.push_back((int)all_corners[i].size());
        for (unsigned int j = 0; j < all_corners[i].size(); j++) {
            all_corners_concatenated.push_back(all_corners[i][j]);
            all_ids_concatenated.push_back(all_ids[i][j]);
        }
    }
    // ��������� ������
    reprojection_error = cv::aruco::calibrateCameraAruco(all_corners_concatenated, all_ids_concatenated,
        marker_counter_per_frame, board, image_size, cameraMatrix,
        distCoeffs, rvecs, tvecs, callibration_flags);
    //��������� ���������� ��������� ������
    bool saveOk = saveCameraParams(output_file, image_size, aspect_ratio, flag, cameraMatrix, distCoeffs, reprojection_error);
    if (!saveOk) {
        std::cerr << "Cannot save output file" << std::endl;
        return;
    }
    //������� ������ ����������
    std::cout << "Rep Error: " << reprojection_error << std::endl;
    //std::cout << "Calibration saved to " << output_file << std::endl;
    return;
}

static bool readDetectorParameters(std::string filename, cv::aruco::DetectorParameters& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params.adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params.adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params.adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params.minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params.minMarkerDistanceRate;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;
    return true;
}

void detectMarker() {   //���������� �������, �� ������� ����� ����� ������� ����
    cv::VideoCapture video(1);
    video.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    video.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    video.set(cv::CAP_PROP_FPS, 30);
    cv::String filename = "params.yml";
    cv::String cam_filename = "calibration.xml";
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    cv::aruco::DetectorParameters detector_params;
    cv::Ptr<cv::aruco::DetectorParameters> detector_ptr;
    readDetectorParameters(filename, detector_params);
    detector_ptr = &detector_params;

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Vec3d> rvecs, tvecs; //������ ��������� ������� ��������� ������ ������������ ������� ��������� �������
    readCameraParameters(cam_filename, camera_matrix, dist_coeffs);
    while (video.grab()) {  //���� ��������, ���� ����� �������� ���� �� ������
        cv::Mat image, image_copy;
        video.retrieve(image);
        image.copyTo(image_copy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        cv::aruco::detectMarkers(image_copy, dictionary, corners, ids, detector_ptr, rejected);
        if (ids.size() > 0) {
            std::cout << corners[0][0] << "\t" << corners[0][1] << "\t" << corners[0][2] << "\t" << corners[0][3] << std::endl;
            //cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
            cv::aruco::estimatePoseSingleMarkers(corners, 23, camera_matrix, dist_coeffs, rvecs, tvecs); //������� ������� �������� � �����������
            drawCubes(image_copy, rvecs, tvecs, camera_matrix, dist_coeffs);    //������ ���
        }
        cv::imshow("out", image_copy);
        char key = (char)cv::waitKey(10);
        if (key == 27) break;
    }
}

void drawCubes(cv::Mat& image, const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    const int marker_size = 23;
    std::vector<cv::Point3d> cube_points;   //������ ����� ���� � ������� ��������� �������
    cube_points.push_back(cv::Point3d(marker_size / 2, marker_size / 2, 0));
    cube_points.push_back(cv::Point3d(marker_size / 2, -marker_size / 2, 0));
    cube_points.push_back(cv::Point3d(-marker_size / 2, -marker_size / 2, 0));
    cube_points.push_back(cv::Point3d(-marker_size / 2, marker_size / 2, 0));
    cube_points.push_back(cv::Point3d(marker_size / 2, marker_size / 2, marker_size));
    cube_points.push_back(cv::Point3d(marker_size / 2, -marker_size / 2, marker_size));
    cube_points.push_back(cv::Point3d(-marker_size / 2, -marker_size / 2, marker_size));
    cube_points.push_back(cv::Point3d(-marker_size / 2, marker_size / 2, marker_size));
    std::vector<cv::Point2d> image_points;
    for (size_t i = 0; i < tvecs.size(); i++) { //���������� ����� ������������� ���������� ������������ ��������
        //���������� ����� � ������� ��������� ������� �� ����������� � ������ ��� ������ ���������� ����� �������� �������� � ����������
        cv::projectPoints(cube_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs, image_points);
        for (size_t i = 0; i < image_points.size(); i++) { //������ ��� � ������� ������� �����
            cv::line(image, image_points[0], image_points[1], cv::Scalar(255, 0, 0), 2);
            cv::line(image, image_points[0], image_points[3], cv::Scalar(255, 0, 255), 2);
            cv::line(image, image_points[0], image_points[4], cv::Scalar(255, 255, 0), 2);
            cv::line(image, image_points[4], image_points[5], cv::Scalar(0, 255, 255), 2);
            cv::line(image, image_points[4], image_points[7], cv::Scalar(0, 255, 0), 2);
            cv::line(image, image_points[1], image_points[2], cv::Scalar(0, 0, 255), 2);
            cv::line(image, image_points[1], image_points[5], cv::Scalar(200, 0, 200), 2);
            cv::line(image, image_points[2], image_points[3], cv::Scalar(0, 0, 0), 2);
            cv::line(image, image_points[2], image_points[6], cv::Scalar(0, 200, 200), 2);
            cv::line(image, image_points[3], image_points[7], cv::Scalar(200, 200, 0), 2);
            cv::line(image, image_points[6], image_points[7], cv::Scalar(50, 50, 50), 2);
            cv::line(image, image_points[5], image_points[6], cv::Scalar(200, 150, 150), 2);
        }
    }
}
