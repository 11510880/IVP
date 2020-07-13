#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;
#include "labeight.h"
#include "labten.h"
#include "labeleven.h"
#include "courseproject1.h"
int main(){
	Mat picture = imread("C://Users//Administrator//Desktop//Images/noisy_fingerprint.pgm ");///polymersomesheadCT-Vandybuilding_originalspot_shaded_text_imageseptagon_noisy_shadedlenaheaheadCT-Vandygoldhilllarge_septagon_gaussian_noise_mean_0_std_50_added
	imshow("source image", picture);
	//Sobel filterig
	//int weights1[] = {-1,-2,-1,0,0,0,1,2,1};
	//int weights2[] = {-1,0,1,-2,0,2,-1,0,1};

	//Prewitt filtering
	int weights1[] = { -1,-1,-1,0,0,0,1,1,1 };
	int weights2[] = { -1,0,1,-1,0,1,-1,0,1 };

	//Robert filtering
	//int weights1[] = { 0,0,0,0,-1,0,0,0,1 };
	//int weights2[] = { 0,0,0,0,0,-1,0,1,0 };
	/*Mat sobelImg = gradientFilter(picture, weights1, weights2,1);
	imshow("Gradient Img(gx)", sobelImg);
	Mat sobelImg2 = gradientFilter(picture, weights1, weights2, 2);
	imshow("Gradient Img(gy)", sobelImg2);
	Mat sobelImg3 = gradientFilter(picture, weights1, weights2, 3);
	imshow("sobel Img(both directions)", sobelImg3);
	Mat sobelImg4= Mat(sobelImg3.size(), sobelImg3.type(), Scalar::all(0));
	vector<int>vec;
	for (int i = 0; i < sobelImg3.rows; i++) {
		for (int j = 0; j < sobelImg3.cols; j++) {
			vec.push_back(sobelImg3.at<uchar>(i, j));
		}
	}
	sort(vec.begin(), vec.end());
	int threshold = int(0.8*vec[vec.size() - 1]);
	for (int i = 0; i < sobelImg3.rows; i++) {
		for (int j = 0; j < sobelImg3.cols; j++) {
			if (sobelImg3.at<uchar>(i, j) < threshold)
				sobelImg4.at<uchar>(i, j) = 0;
			else
				sobelImg4.at<uchar>(i, j) = 255;
		}
	}
	imshow("threshold edge", sobelImg4);*/

	/*Mat cannyImg = cannyFiltering(picture, 13, 3.0);
	imshow("canny Img", cannyImg);
	Mat LoGImg = LaplacianGaussian(picture,1,0.04);
	imshow("LoG Image", LoGImg);;*/

	/*Mat globalThresholdImg = globalThresholding(picture,10);
	imshow("Global Thresholding", globalThresholdImg);*/


	/*Main function for lab 11*/
	/*double weights[25];
	for (int i = 0; i < 25; i++) {
		weights[i] = 1.0 / 25;
	}
	Mat blurImg = imageBlur(picture, 5, weights);
	imshow("blurred Image", blurImg);
	Mat otsuImg = OtsuMethod(picture);
	imshow("Otsu Threshoding", otsuImg);
	Mat partitionImg = partitionOtsu(picture);
	imshow("partition Otsu", partitionImg);
	Mat movingAverageImg = movingAverageThreshold(picture, 20, 0.5);
	imshow("moving average thresholding", movingAverageImg);
	Mat regionGrowImg = regionGrow(picture);
	imshow("region grow", regionGrowImg)*/
	
	waitKey(0);
	
} 