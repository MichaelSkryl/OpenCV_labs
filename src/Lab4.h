#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <complex>
#include <opencv2/intensity_transform.hpp>


const double PI = 3.141592653589793238463;
enum Flags { X_COORD, Y_COORD };
enum Filters { Sobel_x, Sobel_y, Box, Laplace };
enum Type { low_pass, high_pass };

cv::Mat customDft(const cv::Mat& src, cv::Mat& dst, Flags flag, bool is_inverse);
void dft2d(const cv::Mat& src, cv::Mat& dst, bool is_inverse);
cv::Mat fft(const cv::Mat& src, Flags flag, cv::Mat& magnitude);
cv::Mat recursiveFft(cv::Mat& src);
void fft2d(const cv::Mat& src, cv::Mat& dst);
void krasivSpektr(cv::Mat& magI);
void computeDFT(const cv::Mat& image, cv::Mat& complex, cv::Mat& dst);
void getMagnitude(cv::Mat& real, cv::Mat& imag, cv::Mat& magn);
void imageFiltering(const cv::Mat& src, Filters name);
void getPaddedImage(const cv::Mat& src, cv::Mat& padded_image, int proper_rows, int proper_cols);
void getComplexImage(const cv::Mat& src, cv::Mat& complex);
void showMagnitude(cv::Mat& complex_image, cv::Mat& magnitude, cv::String window_name);
void lowHighPassFilters(const cv::Mat& src, Type filter);
void findInNumberPlate(const cv::Mat& src, const cv::Mat& letter);
void findEyes(const cv::Mat& src, const cv::Mat& eye);
void matchTemplate(const cv::Mat& src, const cv::Mat& temp);