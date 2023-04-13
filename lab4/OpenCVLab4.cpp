#include "src/Lab4.h"


int main() {
    cv::Mat image, image1, plate, letter;
    cv::Mat res;
    cv::Mat magn, complex;
    image = cv::imread("steinbeck.jpg", cv::IMREAD_GRAYSCALE);
    cv::imshow("Original", image);
    plate = cv::imread("portrait.jpg", cv::IMREAD_GRAYSCALE);
    letter = cv::imread("glaza.png", cv::IMREAD_GRAYSCALE);
    cv::imshow("Original", image);
    cv::waitKey(0);
    image1 = image.clone();
    image.convertTo(image, CV_64F);
    double duration = 0;
    cv::namedWindow("spectrum magnitude", cv::WINDOW_AUTOSIZE);
    duration = static_cast<double>(cv::getTickCount());
    dft2d(image, res, false);
    duration = static_cast<double>(cv::getTickCount()) - duration;
    duration /= cv::getTickFrequency();
    std::cout << "Custom DFT time: " << duration << std::endl;
    duration = static_cast<double>(cv::getTickCount());
    fft2d(image, res);
    duration = static_cast<double>(cv::getTickCount()) - duration;
    duration /= cv::getTickFrequency();
    std::cout << "Custom FFT time: " << duration << std::endl;
    duration = static_cast<double>(cv::getTickCount());
    computeDFT(image, complex, magn);
    duration = static_cast<double>(cv::getTickCount()) - duration;
    duration /= cv::getTickFrequency();
    std::cout << "Built-in DFT time: " << duration << std::endl;
    cv::destroyAllWindows();
    cv::imshow("Original", image1);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::destroyWindow("2");
    imageFiltering(image, Sobel_x);
    imageFiltering(image, Sobel_y);
    imageFiltering(image, Box);
    imageFiltering(image, Laplace);
    cv::destroyAllWindows();
    lowHighPassFilters(image, low_pass);
    cv::waitKey(0);
    lowHighPassFilters(image, high_pass);
    cv::destroyAllWindows();
    matchTemplate(plate, letter);
    cv::waitKey(0);
}