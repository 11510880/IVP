#ifndef LABTHREE_H
#define LABTHREE_H
#include<opencv2/opencv.hpp>
using namespace cv;
Mat imageTranslation(Mat srcImage, int xOffset, int yOffset);
Mat imageRotation(Mat srcImage, double degree);
Mat imageShearVertical(Mat srcImage, double scale);
Mat imageShearHorizontal(Mat srcImage, double scale);
Mat imageSmoothing(Mat srcImage, int choice);
Mat imageSharpening(Mat srcImage, int choice);
Mat GammaCorrection(Mat srcImage, double gamma);
Mat histogramEnhancement(Mat srcImage, int choice);
#endif
