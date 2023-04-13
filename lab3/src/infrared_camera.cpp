#include "infrared_camera.h"

void DetectTargets(const cv::Mat& src, cv::Mat& dst) {
	cv::Mat tmp = src.clone();
	cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
	dst.create(tmp.size(), tmp.type());
	cv::blur(tmp, dst, cv::Size(5, 5));		//Сглаживаем для снижения влияния шумов
	cv::threshold(dst, dst, 210, 255, cv::THRESH_BINARY); //Пороговая обработка, порог задается вручную
	std::vector<std::vector<cv::Point>> countours;
	std::vector<cv::Point2i> centers;
	findContours(dst, countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //Находим контуры на бинарном изображении
	for (size_t i = 0; i < countours.size(); i++) {
		centers.push_back(GetCenter(countours[i]));			//Получаем центры масс контуров
	}
	dst = src.clone();
	MarkTargets(dst, centers);							//Рисуем крестики - целеуказания
}

void DetectEngine(const cv::Mat& src, cv::Mat& dst) {
	dst.create(src.size(), CV_8UC1);
	std::vector<cv::Mat> planes;
	cv::split(src, planes);		//Делим исходное изображение на три цветовых канала
	IncreaseIntensity(planes[2]);		//Повышаем интенсивность изображения, равномерно поднимая ее ближе к 255
	cv::blur(planes[2], dst, cv::Size(5, 5)); //Для снижения влияния шумов сглаживаем
	cv::threshold(dst, dst, 200, 255, cv::THRESH_BINARY); //Пороговая обработка
	std::vector<std::vector<cv::Point>> countours;
	std::vector<cv::Point2i> centers;
	findContours(dst, countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //Ищем контуры
	if (countours.size() > 1) {
		int largest = FindLargestCountour(countours);	//Находим наибольший контур
		centers.push_back(GetCenter(countours[largest]));
	} else if (countours.size() != 0) {
		centers.push_back(GetCenter(countours[0]));
	}
	dst = src.clone();
	MarkTargets(dst, centers);		//Целеуказание
}

void IncreaseIntensity(cv::Mat& img) { //Повышает интенсивность изображения
	double max_value = 0;
	cv::minMaxLoc(img, nullptr, &max_value);		//Находим максимальное значение на изображении
	cv::add(img, 255 - static_cast<int>(max_value), img);
}

cv::Point2i GetCenter(const std::vector<cv::Point>& countour) {	//Получаем центр масс контура
	cv::Moments moments = cv::moments(countour);
	cv::Point2i countour_center(static_cast<int>(moments.m10 / moments.m00), static_cast<int>(moments.m01 / moments.m00));
	return countour_center;
}

void MarkTargets(cv::Mat& image, const std::vector<cv::Point2i>& centers) {		//Отмечаем цели
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
		double current_area = cv::contourArea(countours[i]);	//Находим наибольший по площади контур
		if (largest_area < current_area) {
			largest_area = current_area;
			index = i;
		}
	}
	return index;
}