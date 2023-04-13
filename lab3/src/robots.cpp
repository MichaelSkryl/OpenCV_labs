#include "robots.h"
#include "infrared_camera.h"


void FindRobots(const cv::Mat& src, cv::Mat& dst) {
	std::vector<cv::Point2i> lamp_center;
	FindLamp(src, dst, lamp_center); //������� ��������
	cv::Mat temp = dst.clone();
	cv::Mat dst1, dst2;
	cv::Mat red_binary, green_binary, blue_binary;
	cv::cvtColor(temp, temp, cv::COLOR_BGR2HSV);	//��������� � HSV
	cv::inRange(temp, cv::Vec3b(0, 85, 85), cv::Vec3b(10, 255, 255), dst1);	//���������� ����������� �������� ����������� �� H � S
	cv::inRange(temp, cv::Vec3b(170, 85, 85), cv::Vec3b(180, 255, 255), dst2); 
	red_binary = dst1 | dst2; //�������� �������� ����������� ������� �������
	cv::erode(red_binary, red_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(18, 12)));	//������� ���� �����
	cv::dilate(red_binary, red_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(18, 12)));
	std::vector<std::vector<cv::Point>> red_countours, green_countours, blue_countours;
	DetermineTeam(dst, red_binary, red_countours, red);	//������� ������� �������
	cv::inRange(temp, cv::Vec3b(45, 90, 90), cv::Vec3b(79, 255, 255), green_binary); //�������� ������� �������
	cv::erode(green_binary, green_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
	cv::dilate(green_binary, green_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10))); //������� ��������� ��������� ����
	DetermineTeam(dst, green_binary, green_countours, green);
	cv::inRange(temp, cv::Vec3b(80, 100, 100), cv::Vec3b(120, 255, 255), blue_binary);	//�� �� ����� ��� ������� �������
	cv::erode(blue_binary, blue_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
	cv::dilate(blue_binary, blue_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
	DetermineTeam(dst, blue_binary, blue_countours, blue);
	FindClosest(dst, red_countours, lamp_center);		//������� ���������
	FindClosest(dst, green_countours, lamp_center);
	FindClosest(dst, blue_countours, lamp_center);
}

void FindLamp(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2i>& center) {
	cv::Mat temp = dst.clone();
	std::vector<cv::Mat> planes;
	cv::cvtColor(src, temp, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(temp, temp, cv::Size(5, 5), 3);//���������� ��� �������� ������� �����
	cv::threshold(temp, temp, 250, 255, cv::THRESH_BINARY); //��������� ��������� ��� ���������� �������� ����� ��������
	std::vector<std::vector<cv::Point>> countours;
	findContours(temp, countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //������� �������
	//DeleteSmallCountours(countours);
	if (countours.size() > 1) {
		int largest = FindLargestCountour(countours);	//������� ���������� ������ - ��������
		center.push_back(GetCenter(countours[largest]));
	} else if (countours.size() != 0) {
		center.push_back(GetCenter(countours[0]));
	}
	MarkTargets(dst, center);	//�������� ��������
}

void DetermineTeam(cv::Mat& src, cv::Mat& dst, std::vector<std::vector<cv::Point>>& countours, Colors color) {
	findContours(dst, countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);		//������� �������
	//DeleteSmallCountours(countours);
	dst = src.clone();
	switch (color) {	//������� ������� ������ �������
	case red:
		cv::polylines(src, countours, true, cv::Scalar(0, 0, 255), 2);
		break;
	case green:
		cv::polylines(src, countours, true, cv::Scalar(0, 255, 0), 2);
		break;
	case blue:
		cv::polylines(src, countours, true, cv::Scalar(255, 0, 0), 2);
		break;
	}
}

/*void DeleteSmallCountours(std::vector<std::vector<cv::Point>>& countours) {
	for (auto iter = countours.begin(); iter != countours.end(); ) {
		if ((*iter).size() < 50) {
			iter = countours.erase(iter);
		} else {
			iter++;
		}
	}
}*/

void FindClosest(cv::Mat& image, const std::vector<std::vector<cv::Point>>& countours, const std::vector<cv::Point2i>& lamp_center) {
	std::vector<cv::Point2i> centers;
	for (size_t i = 0; i < countours.size(); i++) {	//������� ������ ���� �������
		centers.push_back(GetCenter(countours[i]));
	}
	//���������� ������� ���������� �� ������ ��������
	long int min_sqr_distance = (centers[0].x - lamp_center[0].x) * (centers[0].x - lamp_center[0].x) + (centers[0].y - lamp_center[0].y) * (centers[0].y - lamp_center[0].y);
	size_t index = 0;
	for (size_t i = 0; i < centers.size(); i++) {	//������� ����������
		long int distance = (centers[i].x - lamp_center[0].x) * (centers[i].x - lamp_center[0].x) + (centers[i].y - lamp_center[0].y) * (centers[i].y - lamp_center[0].y);
		if (distance < min_sqr_distance) {
			min_sqr_distance = distance;
			index = i;
		}
	}
	cv::circle(image, centers[index], 10, cv::Scalar(255, 255, 255), 3);	//������ ���� ��� ��������� �������
}