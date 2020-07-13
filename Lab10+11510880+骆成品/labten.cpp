#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
#define PI 3.14159265
using namespace cv;
using namespace std;


Mat gradientFilter(Mat srcImage, int weights[], int weights2[],int choice){
	//edge detection using gradient method
	//int choice=1, return |gx|
	//int choice=2, return |gy|
	//int choice=3, return |gx|+|gy|
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	srcImage.convertTo(srcImage, CV_16S);
	int r = srcImage.rows;
	int c = srcImage.cols;
	int ksize = 3;
	copyMakeBorder(srcImage, srcImage, ksize/2, ksize/2, ksize/2, ksize/2, BORDER_CONSTANT, Scalar::all(0));
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	Mat dstImage2 = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	int rows = dstImage.rows;
	int cols = dstImage.cols;
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			vector<short>vec;
			vector<short>vec2;
			int counter = -1;
			for (int m = -ksize / 2; m <= ksize / 2; m++) {
				for (int n = -ksize / 2; n <= ksize / 2; n++) {
					counter++;
					vec.push_back(srcImage.at<short>(i + m, j + n)*weights[counter]);
					vec2.push_back(srcImage.at<short>(i + m, j + n)*weights2[counter]);
				}
			}
			int sum  = 0;
			int sum2 = 0;
			for (int k = 0; k < vec.size(); k++) {
				sum +=  vec[k];
				sum2 += vec2[k];
			}
			dstImage.at<short>(i, j) =abs(sum);
			dstImage2.at<short>(i, j) = abs(sum2);
		}
	}
	normalize(dstImage, dstImage, 0, 255, NORM_MINMAX);
	normalize(dstImage2, dstImage2, 0, 255, NORM_MINMAX);
	//dstImage = dstImage + dstImage2;
	dstImage.convertTo(dstImage, CV_8U);
	dstImage2.convertTo(dstImage2, CV_8U);
	Mat dstImage3(dstImage, Rect(ksize / 2, ksize / 2, c, r));
	Mat dstImage4(dstImage2, Rect(ksize / 2, ksize / 2,c, r));
	switch (choice)
	{
	case 1:
		return dstImage3;
	case 2:
		return dstImage4;
	case 3:
		Mat dstImage5 = Mat(dstImage3.size(), dstImage3.type(), Scalar::all(0));
		for (int i = 0; i < dstImage5.rows; i++) {
			for (int j = 0; j < dstImage5.cols; j++) {
				dstImage5.at<uchar>(i, j) = saturate_cast<uchar>(dstImage3.at<uchar>(i, j) + dstImage4.at<uchar>(i, j));

			}
		}
		return dstImage5;
	}
}


Mat cannyFiltering(Mat srcImage, int lowThreshold, double ratio) {
	//canny filtering 
	//lowThreshold represents the lower threshold, highThreshold=ratio*lowThreshold
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage= Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	int ksize = 3;//kernel size
	blur(srcImage, srcImage, Size(13,13));//blur the image
	Canny(srcImage, dstImage, lowThreshold, ratio*lowThreshold, ksize);
	return dstImage;
}

Mat LaplacianGaussian(Mat srcImage,double sigma, double T) {
	//LoG edge detection
	//first apply a nxn Gaussian filter
	//n=2*ceil(sigma)+1
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	int n = int(ceil(sigma*3)) * 2 + 1;
	vector<double>kernel;
	for (int i = -n/2; i <= n/2; i++) {
		for (int j = -n/2; j <= n/2; j++) {
			double value = exp(-double(i*i + j * j) / (2 * sigma*sigma))*(i*i+j*j-2*sigma*sigma)/(pow(2*sigma,4));
			kernel.push_back(value);
		}
	}
	int ksize = n;
	int r = srcImage.rows;
	int c = srcImage.cols;
	copyMakeBorder(srcImage, srcImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_CONSTANT, Scalar::all(0));
	Mat dstImage = Mat(srcImage.size(), CV_64F, Scalar::all(0));
	Mat resultImage= Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	//dstImage.convertTo(dstImage,);
	int rows = dstImage.rows;
	int cols = dstImage.cols;
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			int counter = -1;
			double sum = 0;
			for (int m = -ksize / 2; m <= ksize / 2; m++) {
				for (int n = -ksize / 2; n <= ksize / 2; n++) {
					counter++;
					sum+=srcImage.at<uchar>(i + m, j + n)*kernel[counter];
				}
			}
			dstImage.at<double>(i, j) =sum;
		}
	}
	//normalize(dstImage, dstImage, 0, 255, NORM_MINMAX);
	//dstImage.convertTo(dstImage,CV_8U);
	
	//convolve with Laplacian operator
	/*int laplacianKernel[] = {1,1,1,1,-8,1,1,1,1};
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			int counter = -1;
			int sum = 0;
			for (int m = -ksize / 2; m <= ksize / 2; m++) {
				for (int n = -ksize / 2; n <= ksize / 2; n++) {
					counter++;
					sum += dstImage.at<short>(i + m, j + n)*laplacianKernel[counter];
				}
			}
			dstImage.at<short>(i, j) = sum;
		}
	}*/

	//zero-crossing test
	int max = dstImage.at<double>(0, 0);
	int min = max;
	cout << max << endl;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (dstImage.at<double>(i, j) > max)
				max = dstImage.at<double>(i, j);
			if (dstImage.at<double>(i, j) < min)
				min = dstImage.at<double>(i, j);
		}
	}
	double threshold =T*max;
	cout << max << ":" << threshold << ":"<<min<<endl;
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			if ((dstImage.at<double>(i - 1, j)*dstImage.at<double>(i + 1, j) < 0&&abs(dstImage.at<double>(i - 1, j) - dstImage.at<double>(i + 1, j)) > threshold)
				|| (dstImage.at<double>(i, j + 1)*dstImage.at<double>(i, j - 1) < 0&&abs(dstImage.at<double>(i, j + 1) - dstImage.at<double>(i, j - 1)) > threshold)||
				(dstImage.at<double>(i - 1, j - 1)*dstImage.at<double>(i + 1, j + 1) < 0&& abs(dstImage.at<double>(i - 1, j - 1) - dstImage.at<double>(i + 1, j + 1)) > threshold)
				|| (dstImage.at<double>(i + 1, j - 1)*dstImage.at<double>(i - 1, j + 1) < 0&& abs(dstImage.at<double>(i - 1, j + 1) - dstImage.at<double>(i + 1, j - 1)) > threshold))
			{
								resultImage.at<uchar>(i, j) =255;
				}
			else
					resultImage.at<uchar>(i, j) =0;
		}
	}
	//normalize(dstImage, dstImage, 0, 255, NORM_MINMAX);
	//dstImage.convertTo(dstImage, CV_8U);
	Mat dstImage2(resultImage, Rect(ksize / 2, ksize / 2, c, r));
	return dstImage2;
}

double getThreshold(Mat srcImage,vector<int>srcImage1,vector<int>srcImage2,double threshold,double T0) {
	//return the global threshold
	//sort(srcImage1.begin(), srcImage1.end());
	//sort(srcImage2.begin(), srcImage2.end());
	vector<int>srcImage3;
	vector<int>srcImage4;
	double I1=0;//average intensity of srcImage1
	double I2=0;//average intensity of srcImage2
	for (int i = 0; i < srcImage1.size(); i++) {
		I1 += srcImage1[i] / double(srcImage1.size());
	}
	for (int i = 0; i < srcImage2.size(); i++) {
		I2 += srcImage2[i] / double(srcImage2.size());
	}
	double T = I1 / 2 + I2 / 2;
	if (T - threshold > T0) {
		for (int i = 0; i < srcImage.rows; i++) {
			for (int j = 0; j < srcImage.cols; j++) {
				if (srcImage.at<uchar>(i, j) > T)
					srcImage3.push_back(srcImage.at<uchar>(i, j));
				else
					srcImage4.push_back(srcImage.at<uchar>(i, j));
			}
		}
		return getThreshold(srcImage, srcImage3, srcImage4, T, T0);
	}
	else {
		return T;
	}
}

Mat globalThresholding(Mat srcImage,double T0) {
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	int max = 0;
	int min = 0;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (srcImage.at<uchar>(i, j) > max)
				max = srcImage.at<uchar>(i, j);
			if (srcImage.at<uchar>(i, j) < min)
				min = srcImage.at<uchar>(i, j);
		    }
	}
	double T = (max + min) / 2;
	vector<int>srcImage1;
	vector<int>srcImage2;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (srcImage.at<uchar>(i, j) > T)
				srcImage1.push_back(srcImage.at<uchar>(i, j));
			else
				srcImage2.push_back(srcImage.at<uchar>(i, j));
		}
	}
	double globalT = getThreshold(srcImage,srcImage1, srcImage2, T, T0);
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (srcImage.at<uchar>(i, j) > globalT)
				dstImage.at<uchar>(i,j)=255;
			else
				dstImage.at<uchar>(i, j) = 0;
		}
	}
	return dstImage;
}