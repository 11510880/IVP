#ifndef LABFOUR_H
#define LABFOUR_H
#include<opencv2/opencv.hpp>
using namespace cv;
void shiftDFT(Mat& mag);
Mat DFT(Mat srcImage, int choice);
//void BLPF(Mat &srcImage, int D0, int n);
//Mat ButterworthFiltering(Mat srcImage, int D0, int n);
//void ILPF(Mat &srcImage, int D0);
//Mat IdelLowpassFiltering(Mat srcImage, int D0);
//void GLPF(Mat &srcImage, int D0);
//Mat GaussianFiltering(Mat srcImage, int D0);
//void GHPF(Mat &srcImage, int D0);
//Mat GaussianHighpassFiltering(Mat srcImage, int D0);

#endif

