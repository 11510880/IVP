#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
#include "labfour.h"
#define PI 3.14159265
using namespace cv;
using namespace std;


Mat logTransformation(Mat srcImage,double c){
	//log transformation
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	srcImage.convertTo(srcImage, CV_32F);
	Mat dstImage= Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	srcImage += Scalar::all(1);
	log(srcImage,srcImage);
	srcImage *= c;
	dstImage = srcImage;
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	dstImage *= 255;
	dstImage.convertTo(dstImage, CV_8U);
	return dstImage;
}

Mat gammaTransformation(Mat srcImage, double gamma) {
	//gamma correction
	//c=1
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	srcImage.convertTo(srcImage, CV_32F);
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			dstImage.at<float>(i,j)=pow(double((srcImage.at<float>(i, j)) / 255.0), gamma)*255.0;
		}
	}
	dstImage.convertTo(dstImage, CV_8U);
	return dstImage;
}

void plotCurve(double c) {
	//function of plotting the log transformation curve
	vector<Point2f>points;
	Mat img(256, 256, CV_8U, Scalar::all(0));
	for (int i = 0; i < 256; i++) {
		float j;
		j = c * log(1 + i);
		j = j/(c*log(256));
		Point2f newPoint = Point2f (i,255-j*255);
		points.push_back(newPoint);
	}
	Mat curve(points, true);
	curve.convertTo(curve, CV_32S); //adapt type for polylines
	polylines(img, curve, false, Scalar(255), 2);
	imshow("log transformation curve", img);
}

Mat backGround() {
	//the function to Generate a 256x256 8-bit image with gray levels varying (using a
	//gaussian distribution) in the range[200, 220], insert a 100x100
	//square in the range[80, 100].
	Mat img(256, 256, CV_8U, Scalar::all(0));
	Point center = Point(img.cols / 2, img.rows / 2);
	double D0 = double(sqrt(-(128 * 128 + 128 * 128) / 2 / log(double(200) / 220.0)));//
	double D1 = double(sqrt(-(50 * 50 + 50 * 50) / 2 / log(double(80) / 100.0)));;
	cout << "D0:"<<D0 <<endl;
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			double D= (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			img.at<uchar>(i, j) =int(220*exp(float(-pow(D,2)/(pow(D0,2)*2))));
		}
	}
	for (int i = 128 - 50; i < 128 + 49; i++) {
		for (int j = 128 - 50; j < 128 + 49; j++) {
			double D = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			img.at<uchar>(i, j) = int(100 * exp(float(-pow(D, 2) / (pow(D1, 2) * 2))));
		}
	}
	return img;
}


Mat removeBackground(Mat srcImage) {
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	//srcImage.convertTo(srcImage, CV_32F);
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			if (srcImage.at<uchar>(i, j) > 150)
				dstImage.at<uchar>(i, j) = 0;
			else {
				dstImage.at<uchar>(i, j) = srcImage.at<uchar>(i, j);
			}
		}
	}
	return dstImage;
}


Mat imageBlur(Mat srcImage, int ksize,double weights[]) {
	//ksize is the size of kernel(ksize x ksize)
	//blur the image using a mean mean filter
	//weights is a 
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	int r = srcImage.rows;
	int c = srcImage.cols;
	copyMakeBorder(srcImage, srcImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_CONSTANT, Scalar::all(0));
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	int rows = dstImage.rows;
	int cols = dstImage.cols;
	for (int i = ksize / 2; i < rows - ksize / 2; i++) {
		for (int j = ksize / 2; j < cols - ksize / 2; j++) {
			vector<double>vec;
			int counter=-1;
			for (int m = -ksize / 2; m <= ksize / 2; m++) {
				for (int n = -ksize / 2; n <= ksize / 2; n++) {
					counter++;
					vec.push_back(srcImage.at<uchar>(i + m, j + n)*weights[counter]);
				}
			}
			double sum = 0;
			for (int k = 0; k < vec.size(); k++) {
				sum += vec[k];
			}
			dstImage.at<uchar>(i, j) = int(sum);
		}
	}
	Mat dstImage2(dstImage, Rect(ksize / 2, ksize / 2, r, c));
	return dstImage2;
}


Mat addSaltandPepper(Mat srcImage, double Pa, double Pb) {
	//Pa is the probability of pepper noise
	//Pb is the probability of salt noise
	RNG rng;
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);}
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	int npepper = int(rows*cols*Pa);
	int nsalt = int(rows * cols*Pb);
	Mat dstImage;
	
	//add pepper noise
	for (int n = 0; n < npepper; n++) {
		srcImage.at<uchar>(rng.uniform(0,rows),rng.uniform(0,cols))=2;
	}
	//add salt noise
	for (int n = 0; n < nsalt; n++) {
		srcImage.at<uchar>(rng.uniform(0, rows), rng.uniform(0, cols)) = 253;
	}
	srcImage.copyTo(dstImage);
	return dstImage;
}



Mat medianFiltering(Mat srcImage, int ksize) {
	//ksize= kernel size
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
			dstImage.at<uchar>(i, j) = vec[vec.size() / 2];
			//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
		}

	}
	Mat dstImage2(dstImage, Rect(ksize / 2, ksize / 2, rows - ksize, cols - ksize));
	return dstImage2;
}

Mat histogramPlot(Mat srcImage) {
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

Mat histogramPlot2(Mat srcImage) {
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); };
	vector<double>pixel(256, 0);
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			pixel[srcImage.at<uchar>(i, j)] += 1;
			
		}
	}
	double maximum = *max_element(pixel.begin(), pixel.end());
	double minimum = *min_element(pixel.begin(), pixel.end());
	for (int i = 0; i < 256; i++) {
		//normalize the histogram value to fit the image size
		pixel[i] -= minimum;
		pixel[i]=double(pixel[i])/(maximum-minimum);
		pixel[i] *= 256;
		cout << pixel[i] << endl;
	}
	Mat dstImage(256, 256, CV_8U, Scalar::all(0));
	for (int i = 0; i < 256; i++) {
		line(dstImage, Point(i, 255), Point(i, 256-pixel[i]),Scalar(255));
	}
	return dstImage;
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
	Mat merge1[2] = { temp,temp };
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
			temp.at<float>(i, j) = float(exp(float(-pow(radius, 2) / (pow(D0, 2) * 2))));
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
			temp.at<float>(i, j) = 1 - float(exp(float(-pow(radius, 2) / (pow(D0, 2) * 2))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}

void IHPF(Mat &srcImage, int D0) {
	Mat temp(srcImage.size(), CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double radius;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			//cout << "temp channels" << temp.channels()<<endl;
			if (radius <= D0) {
				temp.at<float>(i, j) = 0;
			}
			else {
				temp.at<float>(i, j) = 1;
			}
		}
	}
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}


void BHPF(Mat &srcImage, int D0, int n) {
	//create Butterworth highpass filter in the frequency domain
	Mat temp(srcImage.rows, srcImage.cols, CV_32F);
	Point center = Point(srcImage.rows / 2, srcImage.cols / 2);
	double radius;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			temp.at<float>(i, j) = 1-float(1 / (1 + pow(radius / D0, double(2 * n))));
		}
	}
	//The filter should be 2D which filter both the magnitude and phase
	//shiftDFT(temp);
	Mat merge1[2] = { temp,temp };
	merge(merge1, 2, srcImage);
}

Mat ButterworthLPF(Mat srcImage, int D0, int n) {
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
	idft(complex, complex);
	split(complex, plane);
	Mat dst;
	normalize(plane[0], dst, 0, 1, CV_MINMAX);
	dst *= 255;
	dst.convertTo(dst, CV_8U);
	return dst;
}

Mat IdealLPF(Mat srcImage, int D0) {
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
	d *= 255;
	d.convertTo(d, CV_8U);
	return d;
}


Mat GaussianLPF(Mat srcImage, int D0) {
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
	dst *= 255;
	dst.convertTo(dst, CV_8U);
	return dst;
}

Mat GaussianHPF(Mat srcImage, int D0) {
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
	dst *= 255;
	dst.convertTo(dst,CV_8U);
	return dst;
}


Mat ButterworthHPF(Mat srcImage, int D0,int n) {
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
	BHPF(filter, D0, n);
	mulSpectrums(complex, filter, complex, 0);
	shiftDFT(complex);
	idft(complex, complex);
	split(complex, plane);
	Mat dst;
	dst = plane[0];
	normalize(plane[0], dst, 0, 1, CV_MINMAX);
	dst *= 255;
	dst.convertTo(dst, CV_8U);
	return dst;
}


Mat IdealHPF(Mat srcImage, int D0) {
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
	dft(complex, complex,DFT_SCALE);
	shiftDFT(complex);
	Mat filter = complex.clone();
	IHPF(filter, D0);
	mulSpectrums(complex, filter, complex, 0);
	shiftDFT(complex);
	idft(complex, complex);
	split(complex, plane);
	Mat dst;
	dst = plane[0];
	normalize(plane[0], dst, 0, 1, CV_MINMAX);
	dst *= 255;
	dst.convertTo(dst, CV_8U);
	return dst;
}

Mat unsharpMasking(Mat srcImage) {
	//unsharp masking
	//first get a low=pass filtered image using GLPF
	Mat LPFImage = GaussianLPF(srcImage, 50);
	srcImage.convertTo(srcImage, CV_32F);
	LPFImage.convertTo(LPFImage, CV_32F);
	Mat HPFImage = srcImage - LPFImage;
	Mat dstImage;
	dstImage = srcImage + HPFImage;
	normalize(HPFImage, HPFImage, 0, 1,CV_MINMAX);
	imshow("HPF IMAGE", HPFImage);
	normalize(dstImage, dstImage, 0, 1, CV_MINMAX);
	dstImage *= 255;
	dstImage.convertTo(dstImage, CV_8U);
	return dstImage;
}