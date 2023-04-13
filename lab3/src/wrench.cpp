#include "wrench.h"

void ValidateKeys(const cv::Mat& src, cv::Mat& tmplt, cv::Mat& dst) {
	cv::Mat temp = src.clone();
	cv::Mat tmplt_temp = tmplt.clone();
	dst = src.clone();
	std::vector<std::vector<cv::Point>> tmplt_countour;		
	std::vector<std::vector<cv::Point>> gk_countours;
	cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);	
	cv::threshold(tmplt_temp, tmplt_temp, 150, 255, cv::THRESH_BINARY);
	cv::findContours(tmplt_temp, tmplt_countour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //Находим контур шаблона
	cv::GaussianBlur(temp, temp, cv::Size(5, 5), 3);
	cv::threshold(temp, temp, 240, 255, cv::THRESH_BINARY_INV);
	cv::findContours(temp, gk_countours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); //Находим контуры на изображении
	for (size_t i = 0; i < gk_countours.size(); i++) { //Проверяем схожесть контуров, выделяем бракованные и целые
		if (cv::matchShapes(gk_countours[i], tmplt_countour[0], cv::CONTOURS_MATCH_I1, 0) < 6.0) {
			cv::polylines(dst, gk_countours[i], true, cv::Scalar(0, 255, 0), 2);
		} else {
			cv::polylines(dst, gk_countours[i], true, cv::Scalar(0, 0, 255), 2);
		}
	}
}