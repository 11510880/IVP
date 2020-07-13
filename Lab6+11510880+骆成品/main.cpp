#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;
#include "imageresizing.h"
#include "labthree.h"
#include "labfour.h"
#include "labsix.h"
int main(){
	Mat picture2 = imread("C://Users//Administrator//Desktop//Images//goldhill.pgm");
	//picture.convertTo(picture, CV_32FC3);
	//picture /=255;
	imshow("Source Image", picture2);
	/*double tTime;
	tTime = double(getTickCount());
	Mat picture2 = PixelReplication(picture, 1.5, 1.5);
	double s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout << "Pixel Replication:" << s << "s" << endl;
	imshow("Pixel Replication", picture2);
	tTime = double(getTickCount());
	Mat picture3 = NearestNeighbor(picture, 1.5, 1.5);
	s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout << "Nearest Neighborhood:" << s << "s" << endl;
	imshow("Nearest Neighbor", picture3);
	tTime = double(getTickCount());
	Mat picture4 = BilinearInterpolation(picture, 1.5, 1.5);
	s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout << "Bilinear Interpolation:" << s << "s" << endl;
	imshow("Bilinear Interpolation", picture4);
	tTime = double(getTickCount());
	Mat picture5 = BicubicInterpolation(picture, 1.5, 1.5);
	s = 1000 * (double(getTickCount()) - tTime) / getTickCount();
	cout << "Bicubic Interpolation:" << s << "s" << endl;
	imshow("Bicubic Interpolation", picture5);
	double tTime;
	tTime = double(getTickCount());
	Mat picture2 = imageTranslation(picture, -50, -50);
	imshow("Image Translation", picture2);

	Mat picture3 = imageRotation(picture, 45);
	imshow("Image Rotation", picture3);

	Mat picture4 = imageShearVertical(picture, 0.3);
	imshow("Image Shear(Vertically)", picture4);

	Mat picture5 = imageShearHorizontal(picture, 0.3);
	imshow("Image Shear(Horizontally)", picture5);

	

	Mat picture7 = imageSmoothing(picture, 2);
	imshow("Average filtering(5x5)", picture7);

	Mat picture8 = imageSmoothing(picture, 3);
	imshow("Median filtering(3x3)", picture8);

	Mat picture9 = imageSmoothing(picture, 4);
	imshow("Median filtering(5x5)", picture9);

	Mat picture15 = imageSmoothing(picture, 5);
	imshow("Binarization filtering(3x3)", picture15);

	Mat picture16 = imageSmoothing(picture, 6);
	imshow("Binarization filtering(5x5)", picture16);
	Mat picture6 = imageSmoothing(picture, 1);
	imshow("Average filtering(3x3)", picture6);
	Mat picture10 = imageSharpening(picture6, 2);
	imshow("Sobel Sharpening", picture10);

	Mat picture11 = imageSharpening(picture6, 1);
	imshow("Laplacian Sharpening", picture11);

	Mat picture12=GammaCorrection(picture,2.5);
	imshow("Gamma correction(0.1)", picture12);

	Mat picture13= histogramEnhancement(picture12, 1);
	imshow("Histogram enhancement(Global)", picture13);

	Mat picture14= histogramEnhancement(picture12, 2);
	imshow("Histogram enhancement(Local)", picture14);
	Mat dftImage3= DFT(picture, 3);
	imshow("Recover image using phase", dftImage3);

	Mat dftImage7= DFT(picture, 4);
	imshow("Recover image using magnitude", dftImage7);

	Mat picture2 = imageRotation(picture, 90);
	imshow("image rotation", picture2);
	Mat dftImage = DFT(picture2,1);
	imshow("DFT manigtude", dftImage);

	Mat dftImage2 =DFT(picture2,2);
	imshow("DFT phase", dftImage2);

	Mat dftImage4 = ButterworthFiltering(picture, 50, 2);
	imshow("Butterworth LPF", dftImage4);

	Mat dftImage5 = IdelLowpassFiltering(picture, 50);
	imshow("Ideal LPF", dftImage5);

	Mat dftImage6 = GaussianFiltering(picture, 50);
	imshow("Gaussian LPF", dftImage6);

	Mat dftImage8 = GaussianHighpassFiltering(picture, 80);
	//dftImage8.convertTo(dftImage, CV_8U);
	//Butterworth_Low_Paass_Filter(picture);
	imshow("GHPF", dftImage8);
	for (int i = 0; i < dftImage8.rows; i++) {
		for (int j = 0; j < dftImage8.cols; j++) {
			dftImage8.at<float>(i, j) = (dftImage8.at<float>(i, j) > 0.43) ? 1 : 0;
		}
	}
	imshow("Gaussian HPF(threshold)", dftImage8);*/
	Mat homoImage= HomomorphicFiltering(picture2, 100 ,1.0 ,1.25, 0.75);
	imshow("Homomorphic Filtering", homoImage);
	Mat picture = imread("C://Users//Administrator//Desktop//Images//lena.pgm");
	addNoise(picture);
	imshow("Image with noise", picture);
	Mat dftImage = DFT(picture, 1);
	imshow("DFT manigtude(NOISE)", dftImage);
	Mat denoiseImg = bandRejectFiltering(picture,65, 10, 2);
	imshow("denoise image", denoiseImg);
	Mat dftImage2 = DFT(denoiseImg, 1);
	imshow("DFT manigtude2", dftImage2);
	Mat face1 = imread("C://Users//Administrator//Desktop//Images//FACE.PNG");
	imshow("face 1", face1);
	Mat tempLate= imread("C://Users//Administrator//Desktop//Images//template.png");
	imshow("template", tempLate);
	Mat Notface = imread("C://Users//Administrator//Desktop//Images//NOTFACE.PNG");
	imshow("Not a face", Notface);
	Mat matchresult1=templateMatching(tempLate,face1);
	Mat matchresult2 = templateMatching(tempLate,Notface);
	imshow("matching result1", matchresult1);
	imshow("matching result2", matchresult2);
	waitKey(0);
} 