#ifndef COURSEPROJECT1_H
#define COURSEPROJECT1_H
#include<opencv2/opencv.hpp>
using namespace cv;
Mat logTransformation(Mat srcImage, double c);
Mat gammaTransformation(Mat srcImage, double gamma);
void plotCurve(double c);
Mat backGround();
Mat removeBackground(Mat srcImage);
Mat imageBlur(Mat srcImage, int ksize, double weights[]);
Mat addSaltandPepper(Mat srcImage, double Pa, double Pb);
Mat addGaussianNoise(Mat srcImage, int mean, int sigma);
Mat medianFiltering(Mat srcImage, int ksize);
Mat histogramPlot(Mat srcImage);
Mat histogramPlot2(Mat srcImage);
void BLPF(Mat &srcImage, int D0, int n);
void ILPF(Mat &srcImage, int D0);
void GLPF(Mat &srcImage, int D0);
void GHPF(Mat &srcImage, int D0);
void IHPF(Mat &srcImage, int D0);
void BHPF(Mat &srcImage, int D0, int n);
Mat ButterworthLPF(Mat srcImage, int D0, int n);
Mat IdealLPF(Mat srcImage, int D0);
Mat GaussianLPF(Mat srcImage, int D0);
Mat GaussianHPF(Mat srcImage, int D0);
Mat ButterworthHPF(Mat srcImage, int D0, int n);
Mat IdealHPF(Mat srcImage, int D0);
Mat unsharpMasking(Mat srcImage);
#endif
