#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

void DetectTargets(const cv::Mat& src, cv::Mat& dst);
cv::Point2i GetCenter(const std::vector<cv::Point>& countour);
//void DeleteSmallCountours(std::vector<std::vector<cv::Point>>& countours);
void MarkTargets(cv::Mat& image, const std::vector<cv::Point2i>& centers);
void DetectEngine(const cv::Mat& src, cv::Mat& dst);
int FindLargestCountour(const std::vector<std::vector<cv::Point>>& countours);
void IncreaseIntensity(cv::Mat& img);