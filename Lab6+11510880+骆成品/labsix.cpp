#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
#include "labfour.h"
#define PI 3.14159265
using namespace cv;
using namespace std;


void HomomorphicFilter(Mat &srcImage, int D0, double c, double gammaH, double gammaL) {
	Mat temp(srcImage.size(), CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double D;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			D = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			//cout << "temp channels" << temp.channels()<<endl;
				temp.at<float>(i, j) = float((gammaH-gammaL)*(1-exp(-c*D*D/(D0*D0)))+gammaL);
			//temp.at<float>(i, j) = float(exp(float(-pow(D, 2) / (pow(D0, 2) * 2))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}

Mat HomomorphicFiltering(Mat &srcImage, int D0, double c, double gammaH, double gammaL) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);
	srcImage.convertTo(srcImage, CV_32F);
	//srcImage /= 255;
	srcImage += 1;
	
	log(srcImage,srcImage);
	
	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);
	cout << complex.channels() << endl;
	//imshow("complex image", complex);
	dft(complex, complex,DFT_SCALE);
	shiftDFT(complex);
	Mat filter = complex.clone();
	HomomorphicFilter(filter, D0, c, gammaH, gammaL);
	mulSpectrums(complex, filter, complex, 0);
	shiftDFT(complex);
	idft(complex, complex);
	exp(complex, complex);
	split(complex, plane);
	plane[0] -= 1;
	double min, max;
	minMaxLoc(plane[0], &min, &max);
	cout << "min:" << min << "max:" << max << endl;
	Mat dst;
	normalize(plane[0], dst, 0,1, CV_MINMAX);
	//dst *= 255;
	//dst.convertTo(dst, CV_8U);
	return dst;
}

void addNoise(Mat& srcImage) {
	//add sinusoidal noise to the original image
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	srcImage.convertTo(srcImage, CV_32F);
	srcImage /= 255;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			srcImage.at<float>(i,j) += 30*float(cos(300*(i))+cos(300*j))/255;
		}
	}
	normalize(srcImage, srcImage, 0, 1, CV_MINMAX);
	srcImage *= 255;
	srcImage.convertTo(srcImage, CV_8U);
	/*double min, max;
	minMaxLoc(srcImage, &min, &max);
	cout << "min:" << min << "max:" << max << endl;
	//srcImage *= 255;
	//srcImage.convertTo(srcImage, CV_8U);
	normalize(srcImage,srcImage, 0, 1, CV_MINMAX);
	minMaxLoc(srcImage, &min, &max);
	cout << "min:" << min << "max:" << max << endl;
	srcImage *= 255;
	minMaxLoc(srcImage, &min, &max);
	cout << "min:" << min << "max:" << max << endl;
	srcImage.convertTo(srcImage, CV_8U);
	minMaxLoc(srcImage, &min, &max);
	cout << "min:" << min << "max:" << max << endl;*/
}


void bandRejectFilter(Mat &srcImage, double D0, double W, int n) {
	//butterworth bandreject filter
	Mat temp(srcImage.size(), CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double D;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			D = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			//cout << "temp channels" << temp.channels()<<endl;
			temp.at<float>(i, j) = float(1/(1+pow(D*W/(D*D-D0*D0),2*n)));
			//temp.at<float>(i, j) = float(exp(float(-pow(D, 2) / (pow(D0, 2) * 2))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}


Mat bandRejectFiltering(Mat srcImage,double D0,double W,int n) {
	//butterworth bandreject filter
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);
	//srcImage.convertTo(srcImage, CV_32F);
	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);
	cout << complex.channels() << endl;
	//imshow("complex image", complex);
	dft(complex, complex,DFT_SCALE);
	shiftDFT(complex);
	Mat filter = complex.clone();
	//HomomorphicFilter(filter, D0, c, gammaH, gammaL);
	bandRejectFilter(filter, D0, W, n);//construct the filter
	mulSpectrums(complex, filter, complex, 0);//multiply the filter with the image
	shiftDFT(complex);
	idft(complex, complex);
	//exp(complex, complex);
	split(complex, plane);
	//plane[0] -= 1;
	Mat dst;
	dst = plane[0];
	//exp(plane[0], plane[0]);
	double min, max;
	/*minMaxLoc(srcImage, &min, &max);
	cout << "min:" << min << "max:" << max << endl;*/
	normalize(dst, dst, 0, 1, CV_MINMAX);
	dst *= 255;
	dst.convertTo(dst, CV_8U);
	return dst;	
}

Mat templateMatching(Mat tempLate, Mat srcImage) {
	//template matching using image correlation
	//Firstly, we pad the template to the same size as the srcImage
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	if (tempLate.channels() != 1) { cvtColor(tempLate, tempLate, COLOR_BGR2GRAY); };
	int rows1 = tempLate.rows;
	int cols1 = tempLate.cols;
	int rows2 = srcImage.rows;
	int cols2 = srcImage.cols;
	//rows1 = getOptimalDFTSize(rows1);
	//cols1 = getOptimalDFTSize(cols1);
	
	int Rows2 = getOptimalDFTSize(rows2);
	int Cols2 = getOptimalDFTSize(cols2);

	copyMakeBorder(tempLate, tempLate, 0, 2*Rows2-rows1, 0,2*Cols2-cols1 , BORDER_CONSTANT, Scalar::all(0));
	//imshow("template padding", tempLate);
	copyMakeBorder(srcImage, srcImage, 0, 2*Rows2-rows2, 0,2*Cols2-cols2, BORDER_CONSTANT, Scalar::all(0));
	//imshow("srcImage padding", srcImage);
	Mat dstImage;
	
	Mat plane1[]= { Mat_<float>(tempLate),Mat::zeros(tempLate.size(),CV_32F) };
	Mat plane2[] = { Mat_<float>(srcImage),Mat::zeros(srcImage.size(),CV_32F) };
	Mat complex1, complex2;
	merge(plane1, 2, complex1);
	merge(plane2, 2, complex2);
	dft(complex1, complex1);
	dft(complex2, complex2);
	shiftDFT(complex1);
	shiftDFT(complex2);
	Mat complex;
	//now we are going to multiply one DFT with the other's conjugate
	mulSpectrums(complex2, complex1, complex, 0,true);// set the last para as true means take the conjugate of the second para
	shiftDFT(complex);
	idft(complex, complex);
	//exp(complex, complex);
	Mat plane[2];
	split(complex, plane);
	//plane[0] -= 1;
	Mat dst;
	dst = plane[0];
	double min, max;
	minMaxLoc(dst, &min, &max);
	cout << "min:" << min << "max:" << max << endl;
	//exp(plane[0], plane[0]);
	dst -= min;
	dst *= 255 / (max - min);
	//normalize(dst, dst, 0, 1, CV_MINMAX);
	minMaxLoc(dst, &min, &max);
	cout << "min:" << min << "max:" << max << endl;
	//dst *= 255;
	dst.convertTo(dst, CV_8U);
	//dst=Mat()
	Mat q(dst, Rect(0, 0, Rows2-1,Cols2-1));
	return q;
}