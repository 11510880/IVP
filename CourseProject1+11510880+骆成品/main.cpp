#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;
#include "labfour.h"
#include "labseven.h"
#include "courseproject1.h"
int main(){
	Mat picture = imread("C://Users//Administrator//Desktop//Images//lena.pgm");
	cvtColor(picture, picture, COLOR_BGR2GRAY);
	//picture.convertTo(picture, CV_32FC3);
	//picture /=255;
	imshow("Source Image", picture);
	
	
	//log transformation
	Mat logImg = logTransformation(picture, 20);
	imshow("log transformation", logImg);
	//gamma transformation
	Mat gammaImg = gammaTransformation(picture, 2);
	imshow("gamma=2", gammaImg);
	//plot log transformation
	plotCurve(2);
	//add background and insert the 100x100 image
	Mat backgroundImg = backGround();
	imshow("background image", backgroundImg);
	//histogram plot
	Mat histogramImg=histogramPlot2(backgroundImg);
	imshow("histogram image", histogramImg);
	//background remove
	Mat removeBgImg=removeBackground(backgroundImg);
	imshow("Removebackground image", removeBgImg);
	double weights[49];
	for (int i = 0; i < 49; i++) {
		weights[i] = 1.0 / 49;
	}
	//blur the image
	Mat blurImage = imageBlur(picture, 7, weights);
	imshow("blur image",blurImage);

	//salt&pepper noise adding 测试时请单独测试，否则下面的代码都会有噪声
	/*Mat snpImg = addSaltandPepper(picture, 0.1, 0.1);
	imshow("noisy image", snpImg);
	//histogram plot
	Mat histogram1 = histogramPlot2(snpImg);
	imshow("histogram of snpImage", histogram1);
	//filter the image
	Mat filterImg1 = medianFiltering(snpImg, 5);
	imshow("denoised image", filterImg1);
	//histogram plot
	Mat histogram2 = histogramPlot2(filterImg1);
	imshow("histogram of filterImg1", histogram2);*/

	//gaussian blur
	
	Mat gassianBlur = GaussianLPF(picture,50);
	imshow("Gaussian LPF image", gassianBlur);
	//unsharp masking
	Mat unsharpMaskImg = unsharpMasking(picture);
	imshow("unsharp masking", unsharpMaskImg);

	//IHPF,BHPF AND GHPF
	Mat GHPFImg = GaussianHPF(picture, 50);
	imshow("Gaussian HPF image", GHPFImg);
	Mat IHPFImg = IdealHPF(picture, 50);
	imshow("Ideal HPF image", IHPFImg);
	Mat BHPFImg =ButterworthHPF(picture, 50,6);
	imshow("Butterworth HPF image", BHPFImg);
	waitKey(0);
} 