#ifndef LABSIX_H
#define LABSIX_H
#include<opencv2/opencv.hpp>
using namespace cv;
void HomomorphicFilter(Mat &srcImage, int D0, double c, double gammaH, double gammaL);
Mat HomomorphicFiltering(Mat &srcImage, int D0, double c, double gammaH, double gammaL);
void addNoise(Mat &srcImage);
void bandRejectFilter(Mat &srcImage, double D0, double W, int n);
Mat bandRejectFiltering(Mat srcImage, double D0, double W, int n);
Mat templateMatching(Mat tempLate, Mat srcImage);
#endif
