#ifndef LABSEVEV_H
#define LABSEVEN_H
#include<opencv2/opencv.hpp>
using namespace cv;
void shiftDFT(Mat& mag);
Mat DFT(Mat srcImage, int choice);
Mat drawHistogram(Mat &srcImage);
Mat arithmeticMean(Mat srcImage, int ksize);
Mat geometricMean(Mat srcImage,int ksize);
Mat medianFilter(Mat srcImage,int ksize);
Mat alphatrim(Mat srcImage, int d,int ksize);
int adaptiveProcess(Mat srcImage, int rows, int cols, int kernelsize, int maxkernelsize);
Mat adaptiveMedianFilter(Mat srcImage);
#endif
