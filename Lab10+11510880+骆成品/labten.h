#ifndef LABTEN_H
#define LABTEN_H
#include<opencv2/opencv.hpp>
using namespace cv;
Mat gradientFilter(Mat srcImage, int weights[], int weights2[], int choice);
Mat cannyFiltering(Mat srcImage, int lowThreshold, double ratio);
Mat LaplacianGaussian(Mat srcImage, double sigma, double T);
double getThreshold(Mat srcImage, vector<int>srcImage1, vector<int>srcImage2, double threshold, double T0);
Mat globalThresholding(Mat srcImage, double T0);
#endif
