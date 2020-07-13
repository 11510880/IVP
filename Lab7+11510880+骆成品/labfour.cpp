#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
using namespace cv;
using namespace std;

void shiftDFT(Mat& mag) {
	// DFT shift to relocate the DFT centre
	mag = mag(Rect(0, 0, mag.cols&-2, mag.rows&-2));
	int cx = mag.rows / 2;
	int cy = mag.cols / 2;
	//划分象限
	Mat q0(mag, Rect(0, 0, cy, cx));//top left
	Mat q1(mag, Rect(cy, 0, cy, cx));//top right
	Mat q2(mag, Rect(0, cx, cy, cx));//bottom left
	Mat q3(mag, Rect(cy, cx, cy, cx));//bottom right

	//inorder to center the dft,we need to swap the quadrant(top left and bottom right,etc)
	Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);
}


Mat DFT(Mat srcImage,int choice) {
	//choice=1 return the magnitude image 
	//choice=2 return the phase image
	//choice=3 return the synthesize image using magnitude and phase
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	//cout << srcImage.channels() << endl;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);

	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = {Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);//DFT_SCALE | DFT_COMPLEX_OUTPUT,DFT_SCALE
	dft(complex, complex);
	split(complex, plane);//plane[0]=real dft plane[1]=imag dft
	Mat mag, phase;
	cartToPolar(plane[0], plane[1],mag,phase);
	shiftDFT(mag);
	shiftDFT(phase);
	Mat FT, IFT;
	switch (choice)
	{
	case 1:
		//scale the magnitude image and return
		//log transformation to scale the value
		mag += Scalar::all(1);
		log(mag, mag);
		
		//image normalization
		normalize(mag, mag, 0, 1, CV_MINMAX);
		return mag;
	case 2:
		//scale the phase image and return
		phase += Scalar::all(1);
		log(phase, phase);
		//image normalization
		normalize(phase, phase, 0, 1, CV_MINMAX);
		return phase;
	case 3:
		//construct the image using phase only
		polarToCart(Mat::ones(mag.size(), CV_32F), phase, plane[0], plane[1]);
		merge(plane,2, FT);
		shiftDFT(FT);
		dft(FT, IFT, DFT_INVERSE | DFT_REAL_OUTPUT);	
		normalize(IFT, IFT, 0, 1, CV_MINMAX);
		return IFT;
	case 4:
		//construct the image using magnitude only
		polarToCart(mag, Mat::zeros(mag.size(), CV_32F), plane[0], plane[1]);
		merge(plane, 2, FT);
		shiftDFT(FT);
		dft(FT, IFT, DFT_INVERSE | DFT_REAL_OUTPUT);
		normalize(IFT, IFT, 0, 1, CV_MINMAX);
		return IFT;
	}
	
	
}


void BLPF(Mat &srcImage, int D0, int n) {
	//create Butterworth filter in the frequency domain
	Mat temp(srcImage.rows, srcImage.cols, CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double radius;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			temp.at<float>(i, j) = float(1 / (1 + pow(radius / D0, double(2 * n))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = {temp,temp};
	merge(merge1, 2, srcImage);
}


void ILPF(Mat &srcImage, int D0) {
	Mat temp(srcImage.size(), CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double radius;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			//cout << "temp channels" << temp.channels()<<endl;
			if (radius <= D0) {
				temp.at<float>(i, j) = 1;
			}
			else {
				temp.at<float>(i, j) = 0;
			}
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}


void GLPF(Mat &srcImage, int D0) {
	Mat temp(srcImage.size(), CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double radius;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			//cout << "temp channels" << temp.channels()<<endl;
			temp.at<float>(i, j) =float(exp(float(-pow(radius,2)/(pow(D0,2)*2))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}


void GHPF(Mat &srcImage, int D0) {
	Mat temp(srcImage.size(), CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double radius;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			//cout << "temp channels" << temp.channels()<<endl;
			temp.at<float>(i, j) = 1-float(exp(float(-pow(radius, 2) / (pow(D0, 2) * 2))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}

Mat ButterworthFiltering(Mat srcImage, int D0, int n) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);
	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);
	cout << complex.channels() << endl;
	//imshow("complex image", complex);
	dft(complex, complex);
	shiftDFT(complex);
	Mat filter = complex.clone();
	BLPF(filter, D0, n);
	mulSpectrums(complex, filter, complex, 0);
	shiftDFT(complex);
	idft(complex,complex);
	split(complex, plane);
	Mat dst;
	normalize(plane[0],dst, 0, 1, CV_MINMAX);
	return dst;
}

Mat IdelLowpassFiltering(Mat srcImage, int D0) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);
	cout << "Optimal DFT size:" << rows2 << endl;
	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	/*dstImage.convertTo(dstImage,CV_32F);
	for (int i = 0; i < dstImage.rows; i++) {
		for(int j=0;i<dstImage.cols;j++){
			dstImage.at<float>(i, j) *=float(pow(-1, i + j));
		}
	}*/
	Mat plane[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);
	cout << complex.channels() << endl;
	//shiftDFT(complex);
	
	//imshow("complex image", complex);
	dft(complex, complex, DFT_SCALE | DFT_COMPLEX_OUTPUT);
	shiftDFT(complex);
	Mat filter = complex.clone();
	ILPF(filter, D0);
	mulSpectrums(complex, filter, complex, 0);
	Mat IFT;
	shiftDFT(complex);
	dft(complex, IFT, DFT_INVERSE);
	Mat dst[2];
	split(IFT, dst);
	magnitude(dst[0], dst[1], dst[0]);
	Mat d = dst[0];
	//dst.convertTo(dst, CV_8U);
	normalize(d, d, 0, 1, CV_MINMAX);
	return d;
}


Mat GaussianFiltering(Mat srcImage, int D0) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);
	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);
	cout << complex.channels() << endl;
	//imshow("complex image", complex);
	dft(complex, complex);
	shiftDFT(complex);
	Mat filter = complex.clone();
	GLPF(filter, D0);
	mulSpectrums(complex, filter, complex, 0);
	shiftDFT(complex);
	idft(complex, complex);
	split(complex, plane);
	Mat dst;
	normalize(plane[0], dst, 0, 1, CV_MINMAX);
	return dst;
}

Mat GaussianHighpassFiltering(Mat srcImage, int D0) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	//optimize size to implement dft
	int rows2 = getOptimalDFTSize(rows);
	int cols2 = getOptimalDFTSize(cols);
	copyMakeBorder(srcImage, dstImage, 0, rows2 - rows, 0, cols2 - cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complex;
	merge(plane, 2, complex);
	cout << complex.channels() << endl;
	//imshow("complex image", complex);
	dft(complex, complex);
	shiftDFT(complex);
	Mat filter = complex.clone();
	GHPF(filter, D0);
	mulSpectrums(complex, filter, complex, 0);
	shiftDFT(complex);
	idft(complex, complex);
	split(complex, plane);
	Mat dst;
	dst = plane[0];
	normalize(plane[0], dst, 0, 1, CV_MINMAX);
	return dst;
}
