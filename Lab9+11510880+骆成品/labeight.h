#ifndef LABEIGHT_H
#define LABEIGHT_H
#include<opencv2/opencv.hpp>
using namespace cv;
Mat imgErosion(Mat srcImage, int sizex, int sizey, int choice);
Mat imgDilation(Mat srcImage, int sizex, int sizey, int choice);
Mat imgClosing(Mat srcImage, int sizex, int sizey, int choice);
Mat imgOpening(Mat srcImage, int sizex, int sizey, int choice);
Mat boundaryExtraction(Mat srcImage, int sizex, int sizey, int choice);
Mat connectedComponent(Mat srcImage);
Mat borderMergeElem(Mat srcImage);
Mat NonOverLapElem(Mat srcImage);
Mat overLapElem(Mat srcImage, Mat nonOverLapImg, Mat boundaryMergeImg);
#endif

