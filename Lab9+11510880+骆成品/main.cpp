#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;
#include "labeight.h"
int main(){
	Mat picture = imread("C://Users//Administrator//Desktop//Images//bubbles_on_black_background.pgm");
	imshow("source image", picture);
	int sizex =3;
	int sizey =3;
	int choice =1;
	Mat erodeImg=imgErosion(picture, sizex, sizey, choice);
	imshow("image erosion", erodeImg);
	Mat openImg = imgOpening(picture, sizex, sizey, choice);
	imshow("image opening", openImg);
	Mat dilateImg = imgDilation(openImg, sizex, sizey, choice);
	imshow("image dilation of opening", dilateImg);
	Mat closeImg = imgClosing(openImg, sizex, sizey, choice);
	imshow("image closing of opening", closeImg);
	
	Mat boundaryImg = boundaryExtraction(picture, sizex, sizey, choice);
	imshow("image boundary", boundaryImg);
	Mat connectCompImg = connectedComponent(picture);
	imshow("connected components", connectCompImg);
	Mat boundaryMergeImg = borderMergeElem(picture);
	imshow("Border Merge Elements", boundaryMergeImg);
	Mat nonOverLapImg = NonOverLapElem(picture);
	imshow("Non-overlapped Elements", nonOverLapImg);
	Mat OverLapImg = overLapElem(picture, nonOverLapImg, boundaryMergeImg);
	imshow("Overlapped Elements", OverLapImg);
	waitKey(0);
} 