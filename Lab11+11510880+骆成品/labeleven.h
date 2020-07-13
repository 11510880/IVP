#ifndef LABELEVEN_H
#define LABELEVEN_H
#include<opencv2/opencv.hpp>
using namespace cv;
Mat OtsuMethod(Mat srcImage);
Mat partitionOtsu(Mat srcImage);
Mat movingAverageThreshold(Mat srcImage, int n, double b);
Mat RegionGrow(Mat src, Point2i pt, int th);
Mat regionGrow(Mat srcImage);
#endif

