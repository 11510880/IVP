#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
#include "labfour.h"
#define PI 3.14159265
using namespace cv;
using namespace std;


Mat drawHistogram(Mat& srcImage) {
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat gray_hist;

	/// Compute the histograms:
	calcHist(&srcImage, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);
	
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8U, Scalar::all(0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	

	/// Draw for each channelScalar(255, 255, 255), 2, 8, 0
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i))), Scalar(255));
		//rectangle(histImage, Point(bin_w*(i - 1), hist_h-1), Point(bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i))),
			//Scalar(255));
	}
	/// Display
	return histImage;
}


Mat arithmeticMean(Mat srcImage,int ksize) {
	//Arithmetic Mean Filter 
	//ksize= kernel size
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	int r = srcImage.rows;
	int c = srcImage.cols;
	copyMakeBorder(srcImage, srcImage, ksize/2, ksize/2, ksize/2, ksize/2, BORDER_CONSTANT, Scalar::all(0));
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	int rows = dstImage.rows;
	int cols = dstImage.cols;
	for (int i = ksize/2; i < rows-ksize/2; i++) {
		for (int j = ksize/2; j < cols-ksize/2; j++) {
			vector<int>vec;
			for (int m = -ksize/2; m <= ksize/2; m++) {
				for (int n = -ksize/2; n <= ksize/2; n++) {
					vec.push_back(srcImage.at<uchar>(i + m, j + n));
				}
			}
			int sum = 0;
			for (int k = 0; k < vec.size(); k++) {
				sum += vec[k];
			}
			sum/= vec.size();
			dstImage.at<uchar>(i, j) = sum;
		}
	}
	Mat dstImage2(dstImage, Rect(ksize/2,ksize/2,r,c));
	return dstImage2;
}

Mat geometricMean(Mat srcImage,int ksize) {
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	int r = srcImage.rows;
	int c = srcImage.cols;
	copyMakeBorder(srcImage, srcImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_CONSTANT, Scalar::all(0));
	srcImage.convertTo(srcImage, CV_32F);
	srcImage += 1;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	int rows = dstImage.rows;
	int cols = dstImage.cols;
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			float s=1;
			for (int m = -ksize/2; m <= ksize/2; m++) {
				for (int n = -ksize/2; n <= ksize/2; n++) {
					s*=srcImage.at<float>(i + m, j + n);
				}
			}
			dstImage.at<float>(i, j) = float(pow(double(s), double(1./(ksize*ksize))));	
		}
	}

	dstImage -= 1;
	dstImage.convertTo(dstImage, CV_8U);
	Mat dstImage2(dstImage, Rect(ksize/2, ksize/2, r, c));
	cout << dstImage2.at<uchar>(30, 30) << endl;
	return dstImage2;
}


Mat medianFilter(Mat srcImage, int ksize) {
	//ksize= kernel size
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	copyMakeBorder(srcImage, srcImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_CONSTANT, Scalar::all(0));
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = ksize/2; i < rows-ksize/2; i++) {
		for (int j = ksize/2; j < cols-ksize/2; j++) {
					vector<int>vec;
					for (int m = -ksize / 2; m <= ksize / 2; m++) {
						for (int n = -ksize / 2; n<= ksize / 2; n++) {
							//dstImage.at<Vec3b>(i, j)[k] += srcImage.at<Vec3b>(i + m - 2, j + n - 2)[k] / 25;
							vec.push_back(srcImage.at<uchar>(i + m, j + n));
						}
					}
					sort(vec.begin(), vec.end());
					dstImage.at<uchar>(i, j) = vec[vec.size() / 2];
					//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
		}
				
	}
	Mat dstImage2(dstImage, Rect(ksize / 2, ksize / 2, rows - ksize, cols - ksize));
	return dstImage2;
}


Mat alphatrim(Mat srcImage, int d,int ksize) {
	//alpha-trimmed mean filter    
	//delete the d/2 lowest pixels and d/2 highest pixels
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	copyMakeBorder(srcImage, srcImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_CONSTANT, Scalar::all(0));
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			vector<int>vec;
			for (int m = -ksize / 2; m <= ksize / 2; m++) {
				for (int n = -ksize / 2; n <= ksize / 2; n++) {
					//dstImage.at<Vec3b>(i, j)[k] += srcImage.at<Vec3b>(i + m - 2, j + n - 2)[k] / 25;
					vec.push_back(srcImage.at<uchar>(i + m, j + n));
				}
			}
			sort(vec.begin(), vec.end());
			for (int k = d / 2; k <= vec.size() - d / 2 - 1; k++) {
				dstImage.at<uchar>(i, j) += int(double(vec[k]) / (9 - d));
			}
		}
	}
	Mat dstImage2(dstImage, Rect(ksize / 2, ksize / 2, rows - ksize, cols - ksize));
	return dstImage2;
}

int adaptiveProcess(Mat srcImage, int rows, int cols, int kernelsize,int maxkernelsize) {

	//Process of adaptive median filtering 
	vector<int>vec;
	for (int i = -kernelsize / 2;i<= kernelsize/2; i++) {
		for (int j = -kernelsize / 2; j <=kernelsize/2; j++) {
			vec.push_back(srcImage.at<uchar>(rows + i, cols + j));
		}
		sort(vec.begin(),vec.end());
		int A1 = vec[vec.size() / 2] - vec[0];
		int A2 = vec[vec.size() / 2] - vec[vec.size()-1];
		if (A1 > 0 && A2 < 0) {
			//process B
			int B1 = srcImage.at<uchar>(rows , cols) - vec[0];
			int B2= srcImage.at<uchar>(rows, cols ) - vec[vec.size()-1];
			if (B1 > 0 && B2 < 0) {
				return srcImage.at<uchar>(rows , cols );
			}
			else
				return vec[vec.size() / 2];
		}
		else {
			//process A
			kernelsize += 2;
			if (kernelsize <= maxkernelsize) {
				return adaptiveProcess(srcImage,  rows,  cols,  kernelsize, maxkernelsize);
			}
			else
				return vec[vec.size() / 2];
		}
	}
}



Mat adaptiveMedianFilter(Mat srcImage) {
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	int kernelsize = 3;
	int maxsize = 9;//max size of the filter is 9x9
	//add the border
	copyMakeBorder(srcImage, srcImage, maxsize/2, maxsize/2, maxsize/2, maxsize/2, BorderTypes::BORDER_REFLECT);
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = maxsize/2; i <=rows - maxsize / 2 -1; i++) {
		for (int j = maxsize/2; j <=cols-maxsize / 2-1; j++) {
			dstImage.at<uchar>(i, j) = adaptiveProcess(srcImage,i,j,kernelsize,maxsize);
		}
	}
	Mat q(dstImage, Rect(maxsize/2, maxsize/2, rows-maxsize,cols-maxsize));
	return q;
}
