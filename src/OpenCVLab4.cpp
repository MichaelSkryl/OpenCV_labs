#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <complex>
#include <opencv2/intensity_transform.hpp>

const double PI = 3.141592653589793238463;
enum Flags {X_COORD, Y_COORD};
enum Filters {Sobel_x, Sobel_y, Box, Laplace};
enum Type {low_pass, high_pass};

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


int main() {
    cv::Mat image, image1, plate, letter;
    cv::Mat res;
    cv::Mat magn, complex;
    image = cv::imread("steinbeck.jpg", cv::IMREAD_GRAYSCALE);
    cv::imshow("Original", image);
    plate = cv::imread("plate2.jpg", cv::IMREAD_GRAYSCALE);
    letter = cv::imread("Letter2.png", cv::IMREAD_GRAYSCALE);
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

void computeDFT(const cv::Mat& image, cv::Mat& complex, cv::Mat& magnitude) { //Для реализации встроенной функции
    cv::Mat padded_image;
    int optimal_rows = cv::getOptimalDFTSize(image.rows);  //Изображение приводится к оптимальному размеру
    int optimal_cols = cv::getOptimalDFTSize(image.cols);
    //Добавляется паддинг
    copyMakeBorder(image, padded_image, 0, optimal_rows - image.rows, 0, optimal_cols - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<double>(padded_image), cv::Mat::zeros(padded_image.size(), CV_64F) };
    cv::Mat complex_image;
    merge(planes, 2, complex_image);
    dft(complex_image, complex_image);
    complex = complex_image.clone();
    split(complex_image, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    magnitude = planes[0];
    magnitude += cv::Scalar::all(1);
    log(magnitude, magnitude);
    magnitude = magnitude(cv::Rect(0, 0, magnitude.cols & -2, magnitude.rows & -2));
    krasivSpektr(magnitude); //Переставляются квадранты
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
    cv::imshow("spectrum magnitude", magnitude);
}

void krasivSpektr(cv::Mat& magI) {
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

cv::Mat customDft(const cv::Mat& src, cv::Mat& dst, Flags flag, bool is_inverse) {
    dst = src.clone();
    //Изображения переводятся в тип double, те, что с двумя каналами - под комплексные числа
    cv::Mat result(dst.rows, dst.cols, CV_64FC2);
    cv::Mat magnitude(dst.rows, dst.cols, CV_64FC1);
    cv::Mat phase(dst.rows, dst.cols, CV_64FC1);
    cv::Mat real(dst.rows, dst.cols, CV_64FC1);
    cv::Mat imag(dst.rows, dst.cols, CV_64FC1);
    
    if (is_inverse) {
        result = dst.clone();
    }
    if (flag == Y_COORD) { //Если по Y, то транспонируем изображение
        cv::transpose(dst, dst);
        result = dst.clone();
    }
    std::vector<cv::Mat> channels(2);
    cv::split(result, channels);  //Разделяем двухканальное изображение
    unsigned int counter = 0;
    int i = 0;
    while (i < dst.rows) { //Для прохождения по всем строкам
        double* ptr = dst.ptr<double>(i);
        double* real_ptr = channels[0].ptr<double>(i);
        double* imag_ptr = channels[1].ptr<double>(i);
        double* dst_real_ptr = real.ptr<double>(i); //Указатели на изображения для хранения результата
        double* dst_imag_ptr = imag.ptr<double>(i);
        std::complex<double> temp_sum(0, 0);
        std::complex<double> z;
        for (int j = 0; j < dst.cols; j++) {
            if (is_inverse) {
                z = std::complex<double>(0, 2 * PI * ((double)(counter * j) / (double)dst.cols)); //Показатель степени экспоненты
            } else {
                z = std::complex<double>(0, -2 * PI * ((double)(counter * j) / (double)dst.cols)); //Для прямого преобразования меняем знак
            }
            //Данные условия необходимы только потому, что в случае со второй координатой работаем сразу с комплексными данными, костыль, наверное
            if (flag == Y_COORD) {
                temp_sum += std::complex<double>(real_ptr[j], imag_ptr[j]) * exp(z);
            } else {
                if (is_inverse) {
                    temp_sum += std::complex<double>(real_ptr[j], imag_ptr[j]) * exp(z);
                } else {
                    temp_sum += ptr[j] * exp(z);
                }
            }
        }
        if (is_inverse) { //Если обратное преобразование, делим на количество элементов
            dst_real_ptr[counter] = temp_sum.real() / dst.cols;
            dst_imag_ptr[counter] = temp_sum.imag() / dst.cols;
        } else {
            if (flag == Y_COORD) {
                dst_real_ptr[counter] = temp_sum.real();
                dst_imag_ptr[counter] = temp_sum.imag();
            } else {
                real_ptr[counter] = temp_sum.real();
                imag_ptr[counter] = temp_sum.imag();
            }
        }
        counter++;
        if (counter == channels[0].cols) {
            i++;
            counter = 0;
        }
    }
    if (flag == Y_COORD) { //Если вторая координата, то транспонируем обратно
        channels[0] = real.clone();
        channels[1] = imag.clone();
        cv::transpose(channels[0], channels[0]);
        cv::transpose(channels[1], channels[1]);
    }
    getMagnitude(channels[0], channels[1], magnitude);

    if (is_inverse) {
        if (flag == X_COORD) {
            channels[0] = real.clone();
            channels[1] = imag.clone();
        }
        cv::merge(channels, dst);
        cv::magnitude(channels[0], channels[1], magnitude);
        cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
        return magnitude;
    }
    cv::merge(channels, dst);
    return magnitude;
}

void dft2d(const cv::Mat& src, cv::Mat& dst, bool is_inverse) {
    cv::Mat temp = dst.clone();
    cv::Mat result;
    result = customDft(src, temp, X_COORD, is_inverse);
    result = customDft(temp, temp, Y_COORD, is_inverse);
    dst = temp.clone();
    if (is_inverse) {
        cv::imshow("Inverse", result);
    } else {
        cv::imshow("Magnitude", result);
    }
}

void fft2d(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat fourier;
    cv::Mat magnitude;
    fourier = fft(src, X_COORD, magnitude);
    fourier = fft(fourier, Y_COORD, magnitude);
    dst = magnitude.clone();
    cv::imshow("Magnitude", magnitude);
}

void getMagnitude(cv::Mat& real, cv::Mat& imag, cv::Mat& magn) {
    cv::magnitude(real, imag, magn);
    magn += cv::Scalar::all(1); //Переводим в логарифмический масштаб
    log(magn, magn);
    cv::normalize(magn, magn, 0, 1, cv::NORM_MINMAX);
    krasivSpektr(magn);
}

cv::Mat fft(const cv::Mat& src, Flags flag, cv::Mat& magnitude) {
    cv::Mat_<std::complex<double>> dst(src.rows, src.cols);
    magnitude.create(src.rows, src.cols, CV_64FC1);
    cv::Mat real(src.rows, src.cols, CV_64FC1);
    cv::Mat imag(src.rows, src.cols, CV_64FC1);

    std::vector<cv::Mat> channels(2);
    std::vector<cv::Mat> src_chan(2);
    cv::split(dst, channels);
    if (flag == Y_COORD) { //Для второй координаты копируем каналы исходного изображения, то есть вещественную и мнимую часть
        cv::split(src, src_chan);
        channels[0] = src_chan[0].clone();
        channels[1] = src_chan[1].clone();
    } else { //Если это только первая координата, то мнимую часть заполняем нулями
        channels[0] = src.clone();
        channels[1] = cv::Mat::zeros(src.rows, src.cols, CV_64F);
    }
    cv::merge(channels, dst);
    if (flag == Y_COORD) {
        cv::transpose(dst, dst); //Так же, как раньше, транспонируем для преобразования по Y
    }
    cv::Mat_<std::complex<double>> fourier(0, dst.cols);
    for (int i = 0; i < dst.rows; i++) {
        cv::Mat temp;
        cv::Mat_<std::complex<double>> four_temp(1, dst.cols);
        cv::Mat roi = dst(cv::Rect(0, i, dst.cols, 1)); //Вырезаем из изображения одну строку
        temp = roi.clone();
        four_temp = recursiveFft(temp); //Отправляем строку в рекурсивный алгоритм
        fourier.push_back(four_temp);
    }
    if (flag == Y_COORD) {
        cv::transpose(fourier, fourier); //Обратно транспонируем
    }
    std::vector<cv::Mat> planes(2);
    cv::split(fourier, planes);
    getMagnitude(planes[0], planes[1], magnitude);
    return fourier; //Возвращаем комплексный результат
}

cv::Mat recursiveFft(cv::Mat& src) {
    unsigned int N = src.cols;
    cv::Mat_<std::complex<double>> even_res(src.rows, src.cols);
    cv::Mat_<std::complex<double>> odd_res(src.rows, src.cols);
    cv::Mat_<std::complex<double>> even(src.rows, src.cols / 2); //Для хранения четных индексов
    cv::Mat_<std::complex<double>> odd(src.rows, src.cols / 2);  //Для нечетных

    if (N == 1) { //Если осталась одна компонента, то возвращаем ее
        return src;
    } else {
        std::complex<double> z = std::complex<double>(0, -2 * PI / (double)N);
        std::complex<double> w_k = exp(z); //Вычисляем коэффициент Wk
        std::complex<double> w = std::complex<double>(1, 0);
        std::complex<double>* ptr = src.ptr<std::complex<double>>(0);
        std::complex<double>* even_ptr = even.ptr<std::complex<double>>(0);
        std::complex<double>* odd_ptr = odd.ptr<std::complex<double>>(0);
        for (size_t i = 0; i < (N / 2); i++) { //Разделяем на четные и нечетные
            even_ptr[i] = ptr[2 * i];
            odd_ptr[i] = ptr[2 * i + 1];
        }
        even_res = recursiveFft(even); //Запускаем рекурсивный алгоритм для каждой из частей
        odd_res = recursiveFft(odd);

        cv::Mat_<std::complex<double>> result = cv::Mat_<std::complex<double>>::zeros(cv::Size2i(src.cols, src.rows));

        std::complex<double>* result_ptr = result.ptr<std::complex<double>>(0);
        std::complex<double>* even_res_ptr = even_res.ptr<std::complex<double>>(0);
        std::complex<double>* odd_res_ptr = odd_res.ptr<std::complex<double>>(0);
        for (unsigned int i = 0; i < N / 2; i++) { //Получение значений фурье преобразования
            result_ptr[i] = even_res_ptr[i] + w * odd_res_ptr[i];
            result_ptr[i + N/2] = even_res_ptr[i] - w * odd_res_ptr[i];
            w *= w_k;
        }
        return result;
    }
}

void imageFiltering(const cv::Mat& src, Filters name) {
    cv::Mat padded_image, padded_kernel;
    cv::Mat complex_image, complex_kernel;
    cv::Mat kernel;
    double sob_x_kern[3][3] = { {-1, 0, 1},
                                {-2, 0, 2},
                                {-1, 0, 1} };
    double sob_y_kern[3][3] = { { 1, 2, 1},
                                { 0, 0, 0},         //Задаем ядра фильтров
                                {-1,-2,-1} };
    double box_kern[3][3] = { {1, 1, 1},
                              {1, 1, 1},
                              {1, 1, 1} };
    double lap_kern[3][3] = { {0, 1, 0},
                              {1,-4, 1},
                              {0, 1, 0} };

    switch (name) {
    case Sobel_x:
        kernel = cv::Mat(3, 3, CV_64F, sob_x_kern);
        break;
    case Sobel_y:
        kernel = cv::Mat(3, 3, CV_64F, sob_y_kern);
        break;
    case Box:
        kernel = cv::Mat(3, 3, CV_64F, box_kern);
        break;
    case Laplace:
        kernel = cv::Mat(3, 3, CV_64F, lap_kern);
        break;
    default:
        std::cerr << "No filter given" << std::endl;
        break;
    }

    int proper_rows = 0;  
    int proper_cols = 0;
    proper_rows = src.rows + kernel.rows - 1; //Необходимое значение строк и столбцов для предотвращения перехлеста согласно учебнику Гонсалес, Вудс
    proper_cols = src.cols + kernel.cols - 1;
    getPaddedImage(src, padded_image, proper_rows, proper_cols); //Дополняем нулями изображение и фильтр
    cv::copyMakeBorder(kernel, padded_kernel, 0, padded_image.rows - kernel.rows, 0, padded_image.cols - kernel.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    getComplexImage(padded_image, complex_image); //Переводим изображение в вид комплескного, то есть с двумя каналами
    getComplexImage(padded_kernel, complex_kernel);

    cv::Mat image_magn, filter_magn;
    cv::Mat convolution, result;
    dft(complex_image, complex_image); //Фурье образ изображения
    showMagnitude(complex_image, image_magn, "Image magnitude");
    dft(complex_kernel, complex_kernel); //Фурье образ ядра фильтра
    showMagnitude(complex_kernel, filter_magn, "Filter magnitude"); //Отображаем их магнитуды
    cv::mulSpectrums(complex_image, complex_kernel, convolution, 0); //Производим свертку
    dft(convolution, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); //Обратное ДПФ для получения результата

    cv::Mat cropped = result(cv::Rect(0, 0, src.cols, src.rows)).clone(); //Обрезаем изображение до исходного
    if (name == Box) {
        cv::normalize(cropped, cropped, 0, 1, cv::NORM_MINMAX);
    } else {
        cropped.convertTo(cropped, CV_8U);
    }
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Result", 40, 40);
    cv::imshow("Result", cropped);
    cv::waitKey(0);
}

void showMagnitude(cv::Mat& complex_image, cv::Mat& magnitude, cv::String window_name) { //Для отображения магнитуд
    std::vector<cv::Mat> channels(2);
    split(complex_image, channels);
    cv::magnitude(channels[0], channels[1], channels[0]);
    magnitude = channels[0];
    magnitude += cv:: Scalar::all(1);
    log(magnitude, magnitude);
    magnitude = magnitude(cv::Rect(0, 0, magnitude.cols & -2, magnitude.rows & -2));
    krasivSpektr(magnitude);
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window_name, 40, 40);
    cv::imshow(window_name, magnitude);
}

void getComplexImage(const cv::Mat& src, cv::Mat& complex) { //Для перевода в двухканальное
    cv::Mat channels[] = { cv::Mat_<double>(src), cv::Mat::zeros(src.size(), CV_64F) };
    cv::merge(channels, 2, complex);
}

void getPaddedImage(const cv::Mat& src, cv::Mat& padded_image, int proper_rows, int proper_cols) { //Для получения дополненных изображений
    int bottom, right;
    int optimal_rows = cv::getOptimalDFTSize(proper_rows);
    int optimal_cols = cv::getOptimalDFTSize(proper_cols);
    bottom = optimal_rows - src.rows;
    right = optimal_cols - src.cols;
    cv::copyMakeBorder(src, padded_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

void lowHighPassFilters(const cv::Mat& src, Type filter) { //Фильтры высоких/низких частот
    cv::Mat magnitude, complex_image, complex_image_high;
    cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_64FC2);
    cv::Mat result;
    cv::Mat image = src.clone();
    cv::Mat planes[] = { cv::Mat_<double>(image), cv::Mat::zeros(image.size(), CV_64F) };
    merge(planes, 2, complex_image); //Создаем двухканальное изображение
    dft(complex_image, complex_image); //Получаем Фурье-образ
    krasivSpektr(complex_image);
    if (filter == high_pass) { //Обрезаем высокие либо низкие частоты
        cv::circle(complex_image, cv::Point(complex_image.cols / 2, complex_image.rows / 2), 20, cv::Scalar::all(0), -1);
    } else {
        complex_image_high = complex_image.clone();
        cv::circle(complex_image, cv::Point(complex_image.cols / 2, complex_image.rows / 2), 50, cv::Scalar::all(0), -1);
        cv::bitwise_xor(complex_image_high, complex_image, complex_image_high);
        complex_image = complex_image_high.clone();
    }
    std::vector<cv::Mat> channels(2);
    cv::split(complex_image, channels);
    cv::magnitude(channels[0], channels[1], magnitude); //Собираем изображение обратно, отображаем измененную магнитуду
    magnitude += cv::Scalar::all(1);
    log(magnitude, magnitude);
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Filter", magnitude);
    cv::merge(channels, complex_image);
    krasivSpektr(complex_image);
    dft(complex_image, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); //Обратное ДПФ для получения результата
    result.convertTo(result, CV_8U);
    cv::imshow("Output", result);
    cv::waitKey(0);
}

void matchTemplate(const cv::Mat& src, const cv::Mat& temp) { //Для поиска номеров и глаз
    cv::Mat padded_image, padded_template;
    cv::Mat complex_image, complex_template;
    cv::Mat image_sub, template_sub;
    cv::Scalar image_mean, template_mean;
    cv::imshow("Image", src);
    cv::imshow("Template", temp);

    src.convertTo(image_sub, CV_64F);
    temp.convertTo(template_sub, CV_64F);
    image_mean = cv::mean(image_sub); //Вычисляем среднее значение интенсивности для изображения и шаблона
    template_mean = cv::mean(template_sub);
    cv::subtract(image_sub, image_mean, image_sub); //Вычитаем среднее
    cv::subtract(template_sub, template_mean, template_sub);

    int proper_rows = 0;
    int proper_cols = 0;
    proper_rows = src.rows + temp.rows - 1; //Минимально необходимое значение строк и столбцов для предотвращения перехлеста
    proper_cols = src.cols + temp.cols - 1;
    getPaddedImage(image_sub, padded_image, proper_rows, proper_cols); //Дополняем изображения нулями
    cv::copyMakeBorder(template_sub, padded_template, 0, padded_image.rows - temp.rows, 0, padded_image.cols - temp.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    getComplexImage(padded_image, complex_image); //Получаем двухканальные изображения
    getComplexImage(padded_template, complex_template);

    cv::Mat image_magn, filter_magn;
    cv::Mat correlation, result;
    dft(complex_image, complex_image); //Получение Фурье-образов изображения и шаблона
    dft(complex_template, complex_template);
    cv::mulSpectrums(complex_image, complex_template, correlation, 0, true); //Корреляция, флаг в конце - комплексно-сопряженный фурье-образ шаблюона
    dft(correlation, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); //Обратное преобразование для получения результата корреляции

    cv::Mat cropped = result(cv::Rect(0, 0, src.cols, src.rows)).clone(); //Обрезаем до исходных размеров
    cv::normalize(cropped, cropped, 0, 1, cv::NORM_MINMAX);

    double min = 0;
    double max = 0;
    cv::Mat found;
    cv::minMaxLoc(cropped, &min, &max);
    cv::threshold(cropped, found, max - 0.02, max, cv::THRESH_BINARY); //Пороговое разделение для нахождения самых ярких областей - искомых
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Result", 40, 40);
    cv::imshow("Result", cropped);
    cv::imshow("Found", found);
    cv::waitKey(0);
}

void findInNumberPlate(const cv::Mat& src, const cv::Mat& letter) {
    cv::Mat padded_image, padded_letter;
    cv::Mat complex_image, complex_letter;
    int proper_rows = 0;
    int proper_cols = 0;
    proper_rows = src.rows + letter.rows - 1;
    proper_cols = src.cols + letter.cols - 1;
    getPaddedImage(src, padded_image, proper_rows, proper_cols);
    cv::copyMakeBorder(letter, padded_letter, 0, padded_image.rows - letter.rows, 0, padded_image.cols - letter.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    getComplexImage(padded_image, complex_image);
    getComplexImage(padded_letter, complex_letter);

    cv::Mat image_magn, filter_magn;
    cv::Mat correlation, result;
    dft(complex_image, complex_image);
    dft(complex_letter, complex_letter);
    cv::mulSpectrums(complex_image, complex_letter, correlation, 0, true);
    dft(correlation, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat cropped = result(cv::Rect(0, 0, src.cols, src.rows)).clone();
    cv::normalize(cropped, cropped, 0, 1, cv::NORM_MINMAX);

    double min = 0;
    double max = 0;
    cv::Mat found;
    cv::minMaxLoc(cropped, &min, &max);
    cv::threshold(cropped, found,  0.99, 1.0, cv::THRESH_BINARY);
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Result", 40, 40);
    cv::imshow("Result", cropped);
    cv::imshow("Found", found);
    cv::waitKey(0);
}


void findEyes(const cv::Mat& src, const cv::Mat& eye) {
    cv::Mat padded_image, padded_letter;
    cv::Mat complex_image, complex_letter;
    cv::Scalar src_mean, eye_mean;
    cv::Mat src_sub, eye_sub, src_norm, eye_norm;
    src_norm = src.clone();
    eye_norm = eye.clone();
    src_norm.convertTo(src_norm, CV_64F);
    eye_norm.convertTo(eye_norm, CV_64F);
    src_mean = cv::mean(src_norm);
    eye_mean = cv::mean(eye_norm);
    cv::subtract(src_norm, src_mean, src_sub);
    cv::subtract(eye_norm, eye_mean, eye_sub);
    int proper_rows = 0;
    int proper_cols = 0;
    proper_rows = src.rows + eye.rows - 1;
    proper_cols = src.cols + eye.cols - 1;
    getPaddedImage(src_sub, padded_image, proper_rows, proper_cols);
    cv::copyMakeBorder(eye_sub, padded_letter, 0, padded_image.rows - eye.rows, 0, padded_image.cols - eye.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    getComplexImage(padded_image, complex_image);
    getComplexImage(padded_letter, complex_letter);

    cv::Mat image_magn, filter_magn;
    cv::Mat correlation, result;
    dft(complex_image, complex_image);
    dft(complex_letter, complex_letter);
    cv::mulSpectrums(complex_image, complex_letter, correlation, 0, true);
    dft(correlation, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat cropped = result(cv::Rect(0, 0, src.cols, src.rows)).clone();
    cv::normalize(cropped, cropped, 0, 1, cv::NORM_MINMAX);

    double min = 0;
    double max = 0;
    cv::Mat found;
    cv::minMaxLoc(cropped, &min, &max);
    cv::threshold(cropped, found, max - 0.01, max, cv::THRESH_BINARY);
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Result", 40, 40);
    cv::imshow("Result", cropped);
    cv::imshow("Found", found);
    cv::waitKey(0);
}
