#include "infrared_camera.h"

void DetectTargets(const cv::Mat& src, cv::Mat& dst) {
	cv::Mat tmp = src.clone();
	cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
	dst.create(tmp.size(), tmp.type());
	cv::blur(tmp, dst, cv::Size(5, 5));		//���������� ��� �������� ������� �����
	cv::threshold(dst, dst, 210, 255, cv::THRESH_BINARY); //��������� ���������, ����� �������� �������
	std::vector<std::vector<cv::Point>> countours;
	std::vector<cv::Point2i> centers;
	findContours(dst, countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //������� ������� �� �������� �����������
	for (size_t i = 0; i < countours.size(); i++) {
		centers.push_back(GetCenter(countours[i]));			//�������� ������ ���� ��������
	}
	dst = src.clone();
	MarkTargets(dst, centers);							//������ �������� - ������������
}

void DetectEngine(const cv::Mat& src, cv::Mat& dst) {
	dst.create(src.size(), CV_8UC1);
	std::vector<cv::Mat> planes;
	cv::split(src, planes);		//����� �������� ����������� �� ��� �������� ������
	IncreaseIntensity(planes[2]);		//�������� ������������� �����������, ���������� �������� �� ����� � 255
	cv::blur(planes[2], dst, cv::Size(5, 5)); //��� �������� ������� ����� ����������
	cv::threshold(dst, dst, 200, 255, cv::THRESH_BINARY); //��������� ���������
	std::vector<std::vector<cv::Point>> countours;
	std::vector<cv::Point2i> centers;
	findContours(dst, countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //���� �������
	if (countours.size() > 1) {
		int largest = FindLargestCountour(countours);	//������� ���������� ������
		centers.push_back(GetCenter(countours[largest]));
	} else if (countours.size() != 0) {
		centers.push_back(GetCenter(countours[0]));
	}
	dst = src.clone();
	MarkTargets(dst, centers);		//������������
}

void IncreaseIntensity(cv::Mat& img) { //�������� ������������� �����������
	double max_value = 0;
	cv::minMaxLoc(img, nullptr, &max_value);		//������� ������������ �������� �� �����������
	cv::add(img, 255 - static_cast<int>(max_value), img);
}

cv::Point2i GetCenter(const std::vector<cv::Point>& countour) {	//�������� ����� ���� �������
	cv::Moments moments = cv::moments(countour);
	cv::Point2i countour_center(static_cast<int>(moments.m10 / moments.m00), static_cast<int>(moments.m01 / moments.m00));
	return countour_center;
}

void MarkTargets(cv::Mat& image, const std::vector<cv::Point2i>& centers) {		//�������� ����
	if (image.channels() == 1) {
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	}
	for (size_t i = 0; i < centers.size(); i++) {
		image.at<cv::Vec3b>(centers[i].y, centers[i].x) = cv::Vec3b(0, 0, 255);
		cv::line(image, centers[i], cv::Point(centers[i].x + 10, centers[i].y), cv::Scalar(0, 0, 0), 2);
		cv::line(image, centers[i], cv::Point(centers[i].x - 10, centers[i].y), cv::Scalar(0, 0, 0), 2);
		cv::line(image, centers[i], cv::Point(centers[i].x, centers[i].y + 10), cv::Scalar(0, 0, 0), 2);
		cv::line(image, centers[i], cv::Point(centers[i].x, centers[i].y - 10), cv::Scalar(0, 0, 0), 2);
	}
}

int FindLargestCountour(const std::vector<std::vector<cv::Point>>& countours) {
	double largest_area = 0;
	int index = 0;
	for (size_t i = 0; i < countours.size(); i++) {
		double current_area = cv::contourArea(countours[i]);	//������� ���������� �� ������� ������
		if (largest_area < current_area) {
			largest_area = current_area;
			index = i;
		}
	}
	return index;
}