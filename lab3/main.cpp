#include "infrared_camera.h"
#include "robots.h"
#include "wrench.h"
#include<opencv2/videoio.hpp>

using namespace cv;

int main() {
	cv::VideoCapture video("Task1.mp4");
	if (!video.isOpened()) {
		std::cerr << "Video hasn't been opened" << std::endl;
		return 1;
	}
	while (video.grab()) {
		cv::Mat frame, frame_copy;
		video.retrieve(frame);
		frame_copy = frame.clone();
		//FindRobots(frame, frame_copy);
		DetectTargets(frame, frame_copy);
		cv::imshow("Video", frame_copy);
		char c = (char)cv::waitKey(25);
		if (c == 'c') {
			break;
		}
	}
	video.release();
	video.open("robot.mp4");
	while (video.grab()) {
		cv::Mat frame, frame_copy;
		video.retrieve(frame);
		frame_copy = frame.clone();
		FindRobots(frame, frame_copy);
		//DetectTargets(frame, frame_copy);
		cv::imshow("Video", frame_copy);
		char c = (char)cv::waitKey(25);
		if (c == 27) {
			break;
		}
	}
	cv::destroyAllWindows();
	cv::waitKey(0);
	cv::Mat src = cv::imread("teplovizor2.png", IMREAD_UNCHANGED);
	cv::Mat tmplt = cv::imread("gk_tmplt.jpg", IMREAD_GRAYSCALE);
	cv::Mat gk = cv::imread("gk.jpg", IMREAD_UNCHANGED);
	cv::Mat dst = src.clone();
	//DetectTargets(robot, dst);
	ValidateKeys(gk, tmplt, dst);
	cv::imshow("21", dst);
	cv::waitKey(0);
	cv::destroyAllWindows();
	//FindRobots(robot, dst);
	//DetectEngine(src, dst);

	//FindRobots(robot, dst);
	//cv::namedWindow("11", cv::WindowFlags::WINDOW_FREERATIO);
	//cv::namedWindow("22", cv::WindowFlags::WINDOW_FREERATIO);
	//cv::imshow("22", robot);
	return 0;
}