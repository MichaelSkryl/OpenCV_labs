#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

enum Colors { red, green, blue };

void FindRobots(const cv::Mat& src, cv::Mat& dst);
void DetermineTeam(cv::Mat& src, cv::Mat& dst, std::vector<std::vector<cv::Point>>& countours, Colors color);
cv::Point2i GetCenter(const std::vector<cv::Point>& countour);
void DeleteSmallCountours(std::vector<std::vector<cv::Point>>& countours);
void FindLamp(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2i>& center);
void FindClosest(cv::Mat& image, const std::vector<std::vector<cv::Point>>& countours, const std::vector<cv::Point2i>& lamp_center);