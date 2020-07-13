#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
#define PI 3.14159265
using namespace cv;
using namespace std;


Mat imgErosion(Mat srcImage,int sizex,int sizey,int choice) {
	//image Erosion
	//choice=1 rectangular kernel
	//choice=2 cross kernel
	//choice=3 ellipse kernel
	Mat dstImage;
	Mat SE;//structure element
	// we assume that the anchor is at the SE centre
	switch (choice)
	{
	case 1:
		SE = getStructuringElement(MORPH_RECT, Size(sizex, sizey));
		break;
	case 2:
		SE = getStructuringElement(MORPH_CROSS, Size(sizex, sizey));
		break;
	case 3:
		SE = getStructuringElement(MORPH_ELLIPSE, Size(sizex, sizey));
		break;
	}
	erode(srcImage, dstImage, SE);
	return dstImage;
}

Mat imgDilation(Mat srcImage, int sizex, int sizey,int choice) {
	Mat dstImage;
	Mat SE;//structure element
	//we assume that the anchor is at the SE centre
	switch (choice)
	{
	case 1:
		SE = getStructuringElement(MORPH_RECT, Size(sizex, sizey));
		break;
	case 2:
		SE = getStructuringElement(MORPH_CROSS, Size(sizex, sizey));
		break;
	case 3:
		SE = getStructuringElement(MORPH_ELLIPSE, Size(sizex, sizey));
		break;
	}
	dilate(srcImage, dstImage, SE);
	return dstImage;
}

Mat imgOpening(Mat srcImage, int sizex, int sizey,int choice) {
	Mat dstImage;
	Mat SE;//structure element
	switch (choice)
	{
	case 1:
		SE = getStructuringElement(MORPH_RECT, Size(sizex, sizey));
		break;
	case 2:
		SE = getStructuringElement(MORPH_CROSS, Size(sizex, sizey));
		break;
	case 3:
		SE = getStructuringElement(MORPH_ELLIPSE, Size(sizex, sizey));
		break;
	}
	erode(srcImage, dstImage, SE);
	dilate(dstImage, dstImage, SE);
	return dstImage;
}

Mat imgClosing(Mat srcImage, int sizex, int sizey, int choice){
	Mat dstImage;
	Mat SE;//structure element
	switch (choice)
	{
	case 1:
		SE = getStructuringElement(MORPH_RECT, Size(sizex, sizey));
		break;
	case 2:
		SE = getStructuringElement(MORPH_CROSS, Size(sizex, sizey));
		break;
	case 3:
		SE = getStructuringElement(MORPH_ELLIPSE, Size(sizex, sizey));
		break;
	}
	dilate(srcImage, dstImage,SE);
	erode(dstImage, dstImage, SE);
	return dstImage;
}



Mat boundaryExtraction(Mat srcImage, int sizex, int sizey, int choice) {
	//extract the edge of the image 
	//A-A eroded by B, where A is the source image and B is the SE
	Mat dstImage;
	Mat SE;//structure element
	switch (choice)
	{
	case 1:
		SE = getStructuringElement(MORPH_RECT, Size(sizex, sizey));
		break;
	case 2:
		SE = getStructuringElement(MORPH_CROSS, Size(sizex, sizey));
		break;
	case 3:
		SE = getStructuringElement(MORPH_ELLIPSE, Size(sizex, sizey));
		break;
	}
	erode(srcImage, dstImage, SE);
	dstImage = srcImage - dstImage;
	return dstImage;
}


Mat connectedComponent(Mat srcImage) {
	//extract the connected component in an image
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);}
	Mat dstImage, centroids, stats,labels;
	int nLabels = connectedComponentsWithStats(srcImage, dstImage, stats, centroids, 8, CV_32S);
	cout << "Connected component:\t" <<"No. of pixels in connected comp\t" <<"Centroids"<<endl;
	for (int i =1; i < nLabels; i++) {
		//we need to pay attention here. labels[0] represents the background of the image
		//stats 5*nLabels Mat, from left to right:left,top, width, height, area
		//centroids 2xnLabel, represents the location of the centre of each connected component
		cout.width(10);
		cout << i;//the ith connected component
		cout.width(28);
		cout<<stats.at<int>(i, 4);//No. of pixels in a connected components
		cout.width(18);
		cout << "("<<centroids.at<double>(i, 0)<<","<<centroids.at<double>(i, 1)<<")" << endl;//the centre of the connected component
	}
	//now, we are going to map the different components to different colors
	/*dstImage = Mat::zeros(srcImage.size(),CV_8U);
	for (int i = 0; i < dstImage.rows; i++) {
		for (int j = 0; j < dstImage.cols; j++) {
			dstImage.at<uchar>(i,j)=labels.at<>
		}
	}*/
	normalize(dstImage, dstImage, 0, 255, NORM_MINMAX);
	dstImage.convertTo(dstImage, CV_8U);
	return dstImage;
}


Mat borderMergeElem(Mat srcImage) {
	//return the bubbles which is merged with the boundary
	//extract the connected component in an image
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage, centroids, stats, labels;
	Mat dstImage2 = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	int nLabels = connectedComponentsWithStats(srcImage, labels, stats, centroids, 8, CV_32S);
	//Because the first component must contain the white border,so we just return the first connected component
	compare(labels, 1, dstImage, CMP_EQ);
	//As we observer the border cafully, we can see that the bottom border is discontinuous with the other three border, so we need to consider the 
	// last connected component also
	for (int j = 0; j < dstImage2.rows; j++) {
		for (int k = 0; k < dstImage2.cols; k++) {
			dstImage2.at<uchar>(j, k) = saturate_cast<uchar>(dstImage2.at<uchar>(j, k) + dstImage.at<uchar>(j, k));
		}
	}
	compare(labels, 121, dstImage, CMP_EQ);
	for (int j = 0; j < dstImage2.rows; j++) {
		for (int k = 0; k < dstImage2.cols; k++) {
			dstImage2.at<uchar>(j, k) = saturate_cast<uchar>(dstImage2.at<uchar>(j, k) + dstImage.at<uchar>(j, k));
		}
	}
	return dstImage2;
}


Mat NonOverLapElem(Mat srcImage) {
	//return the non-overlapped elements
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	Mat dstImage, centroids, stats, labels;
	int nLabels = connectedComponentsWithStats(srcImage, labels, stats, centroids, 8, CV_32S);
	Mat dstImage2 = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 1; i < nLabels; i++) {
		//traverse all the connected components, only those values between 387-5<x<387+5 would be considered'
		if (stats.at<int>(i, 4) <= 392 && stats.at<int>(i, 4) >= 382) {
			compare(labels, i, dstImage,CMP_EQ);
		
		/*for (int j = 0; j < dstImage2.rows;j++) {
			for (int k = 0; k < dstImage2.cols; k++) {
				dstImage2.at<uchar>(j,k) =saturate_cast<uchar>(dstImage2.at<uchar>(j, k)+dstImage.at<uchar>(j,k));
			}
		}*/
			bitwise_or(dstImage,dstImage2,dstImage2);
		}
	}
	return dstImage2;
}


Mat overLapElem(Mat srcImage, Mat nonOverLapImg, Mat boundaryMergeImg) {
	//return the overlap elements
	//To do this, we just have to subtract the original image with the other two image--non-overlap bubbles and bubbles merged with borders
	if (srcImage.channels() != 1) { cvtColor(srcImage, srcImage, COLOR_BGR2GRAY); }
	if (nonOverLapImg.channels() != 1) { cvtColor(nonOverLapImg, nonOverLapImg, COLOR_BGR2GRAY); }
	if (boundaryMergeImg.channels() != 1) { cvtColor(boundaryMergeImg, boundaryMergeImg, COLOR_BGR2GRAY); }
	Mat mergeImg;
	bitwise_or(nonOverLapImg, boundaryMergeImg, mergeImg);
	//imshow("merge iMAGE", mergeImg);
	Mat dstImg;
	bitwise_not(mergeImg, mergeImg);
	bitwise_and(mergeImg, srcImage, dstImg);
	return dstImg;
}