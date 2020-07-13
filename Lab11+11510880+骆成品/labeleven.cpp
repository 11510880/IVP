#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
#define PI 3.14159265
using namespace cv;
using namespace std;


Mat OtsuMethod(Mat srcImage) {
	//Otsu's Method
	
	//firstly, we have to calculate the histogram of intensity k of level [0,L-1]
	vector<double>pdf(256,0);//Propability density function
	vector<double>cdf(256, 0);//cdf cumulative sum
	
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			pdf[srcImage.at<uchar>(i, j)] += 1.0 / (srcImage.cols*srcImage.rows);
		}
	}
	//secondly, we will obtain the pdf of the image
	for (int i =0; i < 256; i++) {
		for (int j = 0; j <= i; j++) {
			cdf[i] += pdf[j];
		}
	}
	cout << cdf[255] << endl;
	//thirdly, we have to calculate the mean of two classes at each level
	vector<double>means1(256, 0);//mean at each level of C1
	vector<double>means2(256, 0);//mean at each level of C2
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j <= i; j++) {
			if(cdf[i]!=0)
			means1[i] += j * pdf[j] / cdf[i];
		}
	}
	for (int i = 0; i < 255; i++) {
		for (int j = i+1; j <256; j++) {
			if (cdf[i] != 1);
			means2[i] += j*pdf[j] / (1 - cdf[i]);
		}
	}

	//we then calculate the global mean
	double globalmean=0;
	for (int i = 0; i < 256; i++) {
		globalmean += i * pdf[i];
	}
	cout << globalmean << endl;
	//the global variance
	double globalVariance=0;
	for (int i = 0; i < 256; i++) {
		globalVariance += pow((i - globalmean), 2)*pdf[i];
	}
	cout << "globalVariance:"<<globalVariance << endl;
	vector<double>sigmaB(256,0);
	for (int i = 0; i < 256; i++) {
		sigmaB[i] = cdf[i] * pow(means1[i] - globalmean, 2) + (1 - cdf[i])*pow(means2[i] - globalmean, 2);
	}
	
	vector<double>K(256,0);
	for (int i = 0; i < 256; i++) {
		K[i]=sigmaB[i] / globalVariance;
	}
	
	int maximumIndex = max_element(K.begin(), K.end()) - K.begin();
	cout << "maximumIndex:" << maximumIndex << endl;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (srcImage.at<uchar>(i, j) > maximumIndex)
				dstImage.at<uchar>(i, j) =255;
			else
				dstImage.at<uchar>(i, j) = 0;
		}
	}
	return dstImage;
}

Mat partitionOtsu(Mat srcImage) {
	//partition the image into 6 parts and then apply otsu method to each part
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	srcImage.copyTo(dstImage);
	Mat q1= dstImage(Rect(0,0, dstImage.cols/3, dstImage.rows/2));
	Mat q2= dstImage(Rect(dstImage.cols/3, 0, dstImage.cols / 3, dstImage.rows / 2));
	Mat q3= dstImage(Rect(2*dstImage.cols/3, 0, dstImage.cols / 3, dstImage.rows / 2));
	Mat q4 = dstImage(Rect(0, dstImage.rows / 2, dstImage.cols / 3, dstImage.rows / 2));
	Mat q5 = dstImage(Rect(srcImage.cols / 3, srcImage.rows / 2, dstImage.cols / 3, dstImage.rows / 2));
	Mat q6 = dstImage(Rect(2 * srcImage.cols / 3, srcImage.rows/2, dstImage.cols / 3, dstImage.rows / 2));
	Mat q11, q22, q33, q44, q55, q66;
	q11= OtsuMethod(q1);
	q22 = OtsuMethod(q2);
	q33 = OtsuMethod(q3);
	q44 = OtsuMethod(q4);
	q55 = OtsuMethod(q5);
	q66 = OtsuMethod(q6);
	q11.copyTo(q1);
	q22.copyTo(q2);
	q33.copyTo(q3);
	q44.copyTo(q4);
	q55.copyTo(q5);
	q66.copyTo(q6);
	return dstImage;
}


Mat movingAverageThreshold(Mat srcImage, int n, double b) {
	//moving average thresholding
	//n is the number of pixels using in one averaging'
	//T=k*m m is the intensity value at (x,y)
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	//dstImage.convertTo(dstImage, CV_32F);
	//firstly we have to flip the odd number rows to implement a zig-zag scan
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (i % 2 == 1) {
				dstImage.at<uchar>(i, j) = srcImage.at<uchar>(i, srcImage.cols - j - 1);
			}
			else
				dstImage.at<uchar>(i, j) = srcImage.at<uchar>(i, j);
		}
	}
	imshow("reverseimage", dstImage);
	//now, we are going to push the image into a row 
	vector<double>rowImg;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			rowImg.push_back(double(dstImage.at<uchar>(i, j)));
			}
	}

	
	vector<double>M(srcImage.rows*srcImage.cols,0);//average mean value
	for (int i = 0; i < srcImage.rows*srcImage.cols; i++) {
		if (i < n) {
			for (int j = 0; j <=i; j++) {
				M[i] += rowImg[j] / n;
				}
			}
		else {
			for (int j = i - n + 1; j <= i; j++) {
				M[i] += rowImg[j] / n;
			}
		}	
	}
	
	//Now we do the thresholding
	for (int i = 0; i < srcImage.rows*srcImage.cols; i++) {
		if (rowImg[i] > M[i] * b) {
			rowImg[i] = 255.0;
		}
		else
			rowImg[i] = 0.0;
	}
	int counter = -1;
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			counter++;
			dstImage.at<uchar>(i, j) = int(rowImg[counter]);
		}
	}
	//flip the odd number lows back
	Mat dstImage2= Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (i % 2 == 1) {
				dstImage2.at<uchar>(i, j) = dstImage.at<uchar>(i, dstImage.cols - j - 1);
			}
			else
				dstImage2.at<uchar>(i, j) = dstImage.at<uchar>(i, j);
		}
	}
	return dstImage2;
	}

/*void regionGrowing(int row, int col, Mat srcImage) {
	//recursive method for region growing, row and col is the image's rows and cols
	for (int i = row - 1; i <= row + 1; i++)
	{
		for (int j = col - 1; j <= col + 1; j++) {
			if (abs(srcImage.at<uchar>(i, j) - srcImage.at<uchar>(row, col)) <= 68) {
				srcImage.at<uchar>(i,j)= 255;
			}
		}
	}

	for(int i=0;i<)
}*/


Mat RegionGrow(Mat src, Point2i pt, int th)
{

	/************************************************
	source of Code: https://blog.csdn.net/robin__chou/article/details/50071313
	************************************************/
	Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 0;								//生长起点灰度值
	int nCurValue = 0;								//当前生长点灰度值
	Mat matDst = Mat::zeros(src.size(), CV_8UC1);	//创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	 vector<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);			//记录生长点的灰度值

	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.back();						//取出一个生长点
		vcGrowPt.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值

			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) <= th)					//在阈值范围内则生长
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//标记为白色
					vcGrowPt.push_back(ptGrowing);					//将下一个生长点压入栈中
				}
			}
		}
	}
	return matDst.clone();
}

Mat regionGrow(Mat srcImage) {
	//region growing segmentation
	//fisrtly, we have to find the connected components in each image
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage, centroids, stats, labels;
	dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	// we have to threshold the image using T=254
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			if (srcImage.at<uchar>(i, j) > 254) {
				dstImage.at<uchar>(i, j) = 255;
			}
			else
				dstImage.at<uchar>(i, j) = 0;
		}
	}
	imshow("S", dstImage);
	int nLabels = connectedComponentsWithStats(dstImage, labels, stats, centroids, 8, CV_32S);
	//To erode the image with one pixel in each connected component, we just assign the value to the centroid
	Mat dstImage2= Mat(srcImage.size(), srcImage.type(), Scalar::all(0));

	//这部分代码是第四题第一张图的
	Mat regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(1, 0)), int(centroids.at<double>(1, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(13, 0)), int(centroids.at<double>(13, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(12, 0)), int(centroids.at<double>(12, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(11, 0)), int(centroids.at<double>(11, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(10, 0)), int(centroids.at<double>(10, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(9, 0)), int(centroids.at<double>(9, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	//Mat regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(8, 0)), int(centroids.at<double>(8, 1))), 68);
	//bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(7, 0)), int(centroids.at<double>(7, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(6, 0)), int(centroids.at<double>(6, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(5, 0)), int(centroids.at<double>(5, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(4, 0)), int(centroids.at<double>(4, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(3, 0)), int(centroids.at<double>(3, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(2, 0)), int(centroids.at<double>(2, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(14,0)), int(centroids.at<double>(14, 1))), 68);
	bitwise_or(regionImg, dstImage2, dstImage2);
	imshow("bitwise img", dstImage2);
	cout << nLabels-1 << endl;
	/*for (int i = 1; i < nLabels; i++) {
		dstImage.at<uchar>(int(centroids.at<double>(i, 1)), int(centroids.at<double>(i, 0))) =255;
	}*/
	/*Mat regionImg;
	for (int i = 1; i < nLabels; i++) {
		regionImg = RegionGrow(srcImage, Point2i(int(centroids.at<double>(i, 0)), int(centroids.at<double>(i, 1))), 68);
		bitwise_or(regionImg, dstImage2, dstImage2);
	}*/
	//region growing using 8-connectivity
	
	
	return dstImage2;
}