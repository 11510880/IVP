#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;



Mat PixelReplication(Mat srcImage, double k1,double k2) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	int dstRows = int(rows * k1);
	int dstCols = int(cols * k2);
	//创建输出图像矩阵
	Mat dstImage = Mat(Size(dstRows, dstCols), srcImage.type(), Scalar::all(100));
	for (int i = 0; i < dstCols; i++) {
		for (int j = 0; j < dstRows; j++) {
			//防止数组越界
			int ix = cvFloor(i / k1);
			int jy = cvFloor(j / k2);
			if (ix > cols - 1)
				ix = cols - 1;
			if (jy > rows - 1)
				jy = rows - 1;
			for (int k = 0; k < srcImage.channels(); k++) {
				dstImage.at<Vec3b>(j, i)[k] = srcImage.at<Vec3b>(jy, ix)[k];
			}
			
		}
	}
	return dstImage;
}


Mat NearestNeighbor(Mat srcImage, double k1, double k2)
{
	
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	int dstRows =int(rows * k1);
	int dstCols = int(cols * k2);
	//创建输出图像矩阵
	Mat dstImage = Mat(Size(dstRows, dstCols), srcImage.type(), Scalar::all(100));
	for (int i = 0; i < dstCols; i++) {
		for (int j = 0; j < dstRows; j++) {
			//防止数组越界
			int ix = cvRound(i/k1);
			int jy = cvRound(j/k2);
			if (ix > cols - 1)
				ix = cols - 1;
			if (jy > rows - 1)
				jy = rows - 1;
			//mapping
			dstImage.at<Vec3b>(j, i) = srcImage.at<Vec3b>(jy, ix);
		}
	}
	return dstImage;
}


Mat BilinearInterpolation(Mat srcImage, double kx, double ky) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	int dstRows = int(rows * kx);
	int dstCols = int(cols * ky);
	//创建输出图像矩阵
	Mat dstImage = Mat(Size(dstRows, dstCols),CV_8UC3, Scalar::all(0));
	for (int i = 0; i < dstRows; i++)
	{
		//uchar *pDstImage = dstImage.ptr<uchar>(i);
		for (int j = 0; j < dstCols; j++)
		{   

			//求放大图在原图中映射点的下边界
			int ix = int(i/kx);
			int jy = int(j/ky);
			//防止数组越界
			if (ix >= rows - 1)
				ix = rows - 2;
			if (jy >= cols - 1)
				jy = cols - 2;
			double p = abs(i / kx -ix);
			double q = abs(j / ky -jy);
			
			//防止越界
			if (p >= 1)
				p = 1;
			if (q >= 1)
				q = 1;
			for (int k = 0; k < 3; k++) {
				double f1 = (1 - q)*srcImage.at<Vec3b>(ix, jy)[k]+q*srcImage.at<Vec3b>(ix,jy+1)[k];
				double f2 = (1 - q)*srcImage.at<Vec3b>(ix+1, jy)[k] + q * srcImage.at<Vec3b>(ix+1, jy + 1)[k];
				double f = (1 - p)*f1 + p * f2;
				dstImage.at<Vec3b>(i, j)[k] =int(f);
			}
			
			//dstImage.at<Vec3b>(i, j)= (1 - p) * (1 - q) * srcImage.at<Vec3b>(ix, jy) + p * (1 - q)*srcImage.at<Vec3b>(ix, jy + 1) + (1 - p)*q*srcImage.at<Vec3b>(ix + 1, jy)+ p * q* srcImage.at<Vec3b>(ix + 1, jy + 1);
		
			

		}
	}
	return dstImage;
}


//Convolutional Kernel for Bicubic Interpolation
double CubicKernel(double x) {
	double a = -0.5;
	double W;
	if (abs(x) <= 1) {
		W = (a + 2)*pow(abs(x), 3) - (a + 3)*x*x + 1;
	}
	else if (abs(x) > 1 && abs(x) < 2)
	{
		W = a * pow(abs(x), 3) - 5 * a*x*x + 8 * a*abs(x) - 4 * a;
	}
	else
		W = 0;
	return W;
}


Mat BicubicInterpolation(Mat srcImage, double kx, double ky) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	int dstRows = int(rows * kx);
	int dstCols = int(cols * ky);
	Mat dstImage = Mat(Size(dstRows, dstCols), CV_8UC3, Scalar::all(0));
	for (int i = 0; i < dstRows; i++) {
		for (int j = 0; j < dstCols; j++)
		{
			//求映射在原图中的点的下边界
			int ix = int(i/kx);
			int jy = int(j/ky);
			double p = abs(i/kx - ix);
			double q = abs(j/ky - jy);
			int mappingvalue=0;
			for (int u = -1; u < 3; u++) {
				for (int v = -1; v < 3; v++) {
					//求两个方向上的kernel值
					double W1 = CubicKernel(abs(u-q));
					double W2 = CubicKernel(abs(v-p));
					
					//边界做简单的mapping
					if (jy<=0) {
						 jy= 1;
					}
					else if (jy >= cols-2) {
						jy = cols - 3;
					}
					if (ix <= 0) {
						ix=1;
					}
					else if (ix >= rows-2) {
						ix = rows - 3;
					}
					//卷积的实现
					mappingvalue = mappingvalue + int(W1 * W2*srcImage.at<Vec3b>(ix+v,jy+u)[0]);
					
				}
			}
			//防止pixel value 溢出
			for (int k = 0; k < 3; k++) {
				if (mappingvalue < 0)
					mappingvalue = 0;
				else if (mappingvalue > 255)
					mappingvalue = 255;
				//mapping
				dstImage.at<Vec3b>(i, j)[k] = mappingvalue;
			}


		}
	}
	//resize(srcImage, dstImage, dstImage.size(), kx, ky, INTER_CUBIC);
	return dstImage;
}
int main() {
	Mat picture = imread("C://Users//Administrator//Desktop//Images//lena.pgm",IMREAD_COLOR);
	imshow("Source Image", picture);
	double tTime;
	tTime = double(getTickCount());
	Mat picture2 = PixelReplication(picture,1.5,1.5);
	double s = 1000*(double(getTickCount()) - tTime)/getTickCount();
	cout <<"Pixel Replication:"<< s << "s"<<endl;
	imshow("Pixel Replication", picture2);
	tTime = double(getTickCount());
	Mat picture3 = NearestNeighbor(picture, 1.5 ,1.5);
	 s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout <<"Nearest Neighborhood:"<< s << "s" << endl;
	imshow("Nearest Neighbor",picture3);
	tTime = double(getTickCount());
	Mat picture4 = BilinearInterpolation(picture,1.5,1.5);
	 s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout <<"Bilinear Interpolation:"<< s << "s" << endl;
	imshow("Bilinear Interpolation", picture4);
	tTime = double(getTickCount());
	Mat picture5 = BicubicInterpolation(picture, 1.5, 1.5);
	 s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout <<"Bicubic Interpolation:"<< s << "s" << endl;
	imshow("Bicubic Interpolation", picture5);
	waitKey(0);
}