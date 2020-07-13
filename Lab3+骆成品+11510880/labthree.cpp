#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<algorithm>
using namespace cv;
using namespace std;
Mat imageTranslation(Mat srcImage, int xOffset, int yOffset) {
	//xOffset represents the offset horizontally, yOffset represents the offset vertically
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//inverse mapping
			int x = j - xOffset;
			int y = i - yOffset;
			if (x >= 0 && y >= 0 && x < cols&&y < rows)
				for (int k = 0; k < srcImage.channels(); k++) {
					dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(y, x)[k];
				}
		}
	}
	return dstImage;
}

Mat imageRotation(Mat srcImage, double degree) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	double radian = degree * CV_PI / 180;
	double y1 = cols*sin(radian);
	double y2 = rows * cos(radian);
	double y3= rows * cos(radian) + cols * sin(radian);
	double x1 = cols * cos(radian);
	double x2 = -rows * sin(radian);
	double x3 = -rows * sin(radian) + cols * cos(radian);
	double ymin = min(min(min(0.0,y1), y2), y3);
	double ymax= max(max(max(0.0, y1), y2), y3);
	double xmin = min(min(min(0.0, x1), x2),x3);
	double xmax = max(max(max(0.0, x1), x2), x3);
	int nrows = abs(ymax-ymin);
	int ncols = abs(xmax-xmin);
	Mat dstImage = Mat(nrows,ncols, srcImage.type(), Scalar::all(0));
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			//inverse mapping
			
			int y = cvFloor((i+xmin)* cos(radian)+(j+ymin)*sin(radian));
			int x = cvFloor(-(i +xmin) * sin(radian) + (j +ymin)* cos(radian));
			if (x >= 0 && y >= 0 && x < cols&&y < rows)
				for (int k = 0; k < srcImage.channels(); k++) {
					dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(y, x)[k];
				}
		}
	}
	return dstImage;		
}

Mat imageShearVertical(Mat srcImage, double scale) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	double y1 = cols;
	double y2 = rows;
	double x1 =cvFloor(-scale*cols) ;
	double x2 =cvFloor(rows-scale*cols);
	double x3 =cvFloor(rows);
	double ymin = min(min(0.0, y1), y2);
	double ymax = max(max(0.0, y1), y2);
	double xmin = min(min(min(0.0, x1), x2), x3);
	double xmax = max(max(max(0.0, x1), x2), x3);
	int nrows = abs(xmax - xmin);
	int ncols = abs(ymax - ymin);
	Mat dstImage = Mat(nrows,ncols,srcImage.type(), Scalar::all(0));
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			//inverse mapping
			int x = cvFloor(i - scale * j);
			int y = j;
			if (x >= 0 && y >= 0 && x < cols&&y < rows)
				for (int k = 0; k < srcImage.channels(); k++) {
					dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(x, y)[k];
				}
		}
	}
	return dstImage;

};


Mat imageShearHorizontal(Mat srcImage, double scale) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	double x1 = cols;
	double x2 = rows;
	double y1 = cvFloor(-scale * rows);
	double y2 = cvFloor(cols - scale * rows);
	double y3 = cvFloor(cols);
	double xmin = min(min(0.0, x1), x2);
	double xmax = max(max(0.0, x1), x2);
	double ymin = min(min(min(0.0, y1), y2), y3);
	double ymax = max(max(max(0.0, y1), y2), y3);
	int nrows = abs(xmax - xmin);
	int ncols = abs(ymax - ymin);
	Mat dstImage = Mat(nrows, ncols, srcImage.type(), Scalar::all(0));
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			//inverse mapping
			int x = i;
			int y = cvFloor(j - scale * x);
			if (x >= 0 && y >= 0 && x < cols&&y < rows)
				for (int k = 0; k < srcImage.channels(); k++) {
					dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(x, y)[k];
				}
		}
	}
	return dstImage;
};


Mat imageSmoothing(Mat srcImage, int choice) {
	//choice represents the type of filtering:
	//choice=1 --3x3 average filter
	//choice=2 --5x5 average filter
	//choice=3 --3x3 median filter
	//choice=4 --5x5 median filter
	//choice=5 --3x3 bilateral filter
	//choice=6 --5x5 bilateral filter
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	switch (choice)
	{
	case 1:
		//3x3 average filter
		//int kernel[3][3] = { 1,1,1,1,1,1,1,1,1 };
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 1 >= 0) && (i + 1 <= rows - 1) && (j - 1 >= 0) && (j + 1 <= cols - 1))
						dstImage.at<Vec3b>(i, j)[k] = (srcImage.at<Vec3b>(i-1,j-1)[k]+ srcImage.at<Vec3b>(i-1, j)[k]+ srcImage.at<Vec3b>(i-1, j+1)[k]+ srcImage.at<Vec3b>(i, j-1)[k]
							+srcImage.at<Vec3b>(i, j)[k]+ srcImage.at<Vec3b>(i, j+1)[k]+ srcImage.at<Vec3b>(i+1, j-1)[k]+ srcImage.at<Vec3b>(i+1, j)[k]
							+ srcImage.at<Vec3b>(i+1, j+1)[k])/9;
					else 
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		break;
	case 2:
		//int kernel[5][5] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 2 >= 0) && (i + 2 <= rows - 1) && (j - 2 >= 0) && (j + 2 <= cols - 1))
					{
						for (int m = 0; m < 5; m++) {
							for (int n = 0; n < 5; n++) {
								dstImage.at<Vec3b>(i, j)[k]+=srcImage.at<Vec3b>(i+m-2,j+n-2)[k]/25;
							}
						}
						//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
					}
					else
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		break;
	case 3:
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 1 >= 0) && (i + 1 <= rows - 1) && (j - 1 >= 0) && (j + 1 <= cols - 1))
					{	
						vector<int>vec;
						for (int m = 0; m <3; m++) {
							for (int n = 0; n < 3; n++) {
								//dstImage.at<Vec3b>(i, j)[k] += srcImage.at<Vec3b>(i + m - 2, j + n - 2)[k] / 25;
								vec.push_back(srcImage.at<Vec3b>(i+m-1,j+n-1)[1]);
								
							}
						}
						sort(vec.begin(), vec.end());
						dstImage.at<Vec3b>(i, j)[k] = vec[vec.size() / 2];
						//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
					}
					else
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		break;

	case 4:
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 2 >= 0) && (i + 2<= rows - 1) && (j - 2 >= 0) && (j + 2 <= cols - 1))
					{
						vector<int>vec;
						for (int m = 0; m < 5; m++) {
							for (int n = 0; n < 5; n++) {
								//dstImage.at<Vec3b>(i, j)[k] += srcImage.at<Vec3b>(i + m - 2, j + n - 2)[k] / 25;
								vec.push_back(srcImage.at<Vec3b>(i + m - 2, j + n - 2)[1]);

							}
						}
						sort(vec.begin(), vec.end());
						dstImage.at<Vec3b>(i, j)[k] = vec[vec.size() / 2];
						//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
					}
					else
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		break;
	case 5:
		//binarization filtering
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 1 >= 0) && (i + 1 <= rows - 1) && (j - 1 >= 0) && (j + 1 <= cols - 1))
					{
						vector<int>vec;
						for (int m = 0; m < 3; m++) {
							for (int n = 0; n < 3; n++) {
								//dstImage.at<Vec3b>(i, j)[k] += srcImage.at<Vec3b>(i + m - 2, j + n - 2)[k] / 25;
								vec.push_back(srcImage.at<Vec3b>(i + m - 1, j + n - 1)[1]);

							}
						}
						sort(vec.begin(), vec.end());
						int median = vec[vec.size() / 2];
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k] >= median ? 255 : 0;
						//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
					}
					else
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		
		break;
	case 6:
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 2 >= 0) && (i + 2<= rows - 1) && (j - 2 >= 0) && (j + 2 <= cols - 1))
					{
						vector<int>vec;
						for (int m = 0; m < 3; m++) {
							for (int n = 0; n < 3; n++) {
								//dstImage.at<Vec3b>(i, j)[k] += srcImage.at<Vec3b>(i + m - 2, j + n - 2)[k] / 25;
								vec.push_back(srcImage.at<Vec3b>(i + m - 2, j + n - 2)[1]);

							}
						}
						sort(vec.begin(), vec.end());
						int median = vec[vec.size() / 2];
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k] >= median ? 255 : 0;
						//dstImage.at<Vec3b>(i, j)[k]=dstImage.at<Vec3b>(i,j)[k]/25;  
					}
					else
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		break;
	}
	
	return dstImage;
}


Mat imageSharpening(Mat srcImage, int choice) {
	//image sharpening
	//choice=1 Laplacian Operator
	//choice=2 Sobel Operator
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	//this result image takes no consideration of the edge of the input image;
	Mat dstImage2 = Mat(rows - 2, cols - 2, CV_64FC1, Scalar::all(0));
	int laplacian[3][3] = {0,1,0,1,-4,1,0,1,0};
	switch (choice)
	{
	case 1:
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 1 >= 0) && (i + 1 <= rows - 1) && (j - 1 >= 0) && (j + 1 <= cols - 1)) {
						int result = 0;
						for (int m = 0; m < 3; m++) {
							for (int n = 0; n < 3; n++) {
								result += srcImage.at<Vec3b>(i + m - 1, j + n - 1)[0] * laplacian[m][n];
							}
						}
						dstImage.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(srcImage.at<Vec3b>(i, j)[k]- result);
					}
					else
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
				}
			}
		}
		break;
	case 2:
		int sobelx[3][3] = {1,0,-1,
			                2,0,-2,
			                1,0,-1};
		int sobely[3][3] = {1,2,1,
			                0,0,0,
			               -1,-2,-1};
		
		for(int i=1;i<rows-1;i++){
			for (int j = 1; j < cols-1; j++) {
				int gx = 0;
		        int gy = 0;							
						for (int m = -1; m < 2; m++) {
							for (int n = -1; n < 2; n++) {
								 gx += srcImage.at<Vec3b>(i + m, j + n)[1] * sobelx[m+1][n+1];
								 gy += srcImage.at<Vec3b>(i + m, j + n)[1] * sobely[m+1][n+1];
							}
						}
						double g = sqrt(pow(gx, 2) + pow(gy, 2));
						// scale the result
						dstImage2.at<double>(i-1, j-1)=(g>100?255:0);
						//dstImage2.at<double>(i - 1, j - 1) = g;
						imshow("dstImage22", dstImage2);
			}
		}
		//scale the result image to 0-255
		/*double minVal;
		double maxVal;
		minMaxLoc(dstImage2, &minVal, &maxVal);
		cout << minVal << maxVal << endl;
		for (int i = 0; i < rows - 2; i++) {
			for (int j = 0; j < rows - 2; j++)
			{
				dstImage2.at<double>(i, j) -= minVal;
			}
		}
		minMaxLoc(dstImage2, &minVal, &maxVal);
		cout << minVal<<maxVal << endl;
		for (int i = 0; i < rows - 2; i++) {
			for (int j = 0; j < rows - 2; j++)
			{
				dstImage2.at<double>(i, j) *= 255 / maxVal;
			}
		}
		minMaxLoc(dstImage2, &minVal, &maxVal);
		cout << minVal << maxVal << endl;
		//add the result image back*/
		imshow("dstImage2", dstImage2);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if ((i - 1 >= 0) && (i + 1 <= rows - 1) && (j - 1 >= 0) && (j + 1 <= cols - 1))
						dstImage.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(srcImage.at<Vec3b>(i, j)[k] + int(dstImage2.at<double>(i - 1, j - 1)));
						//dstImage.at<Vec3b>(i, j)[k] = (srcImage.at<Vec3b>(i, j)[k] + int(dstImage2.at<double>(i-1, j-1))>255?255: srcImage.at<Vec3b>(i, j)[k] + int(dstImage2.at<double>(i - 1, j - 1)));
					else
					{
						//for simplicity, the edge just keep unchanged
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
					}
				}
			}
		}
		break;
	}
	return dstImage;

}


Mat GammaCorrection(Mat srcImage,double gamma) {
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				dstImage.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(pow(double((srcImage.at<Vec3b>(i, j)[k])/255.0),gamma)*255.0);
			}
		}
	}
	Mat mean;
	Mat stddev;
	Mat gray;
	cvtColor(dstImage, gray, CV_RGB2GRAY);
	meanStdDev(gray, mean, stddev);
	double std = stddev.at<double>(0,0);
	double variance = pow(std, 2);
	cout << "variance:" << variance << endl;
	return dstImage;
}


Mat histogramEnhancement(Mat srcImage,int choice) {
	//choice=1 global enhancement
	//choice=2 local enhancement
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	Mat dstImage = Mat(srcImage.size(), srcImage.type(), Scalar::all(0));
	vector<double>pixel(256, 0);
	vector<int>mapVal(256, 0);
	switch (choice)
	{
	case 1:	
		int k;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				k = srcImage.at<Vec3b>(i, j)[1];
				pixel[k] += 1;
			}
		}
		for (int i = 0; i < 256; i++) {
			pixel[i] /= rows * cols;
		}
		for (int i = 0; i < mapVal.size(); i++) {
			for (int j = 0; j <= i; j++) {
				mapVal[i] += cvRound(255 * pixel[j]);
			}

		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					dstImage.at<Vec3b>(i, j)[k] = mapVal[srcImage.at<Vec3b>(i, j)[1]];
				}
			}
		}
		break;
	case 2:
		//in this case, we would like to enhance the contrast of the dark region
		double k0 = 0.4;
		double k1 = 0.02;
		double k2 = 2;
		double E = 4.0;
		Mat mean;
		Mat std;
		Mat gray;
		cvtColor(srcImage, gray, CV_RGB2GRAY);
		meanStdDev(gray, mean,std);
		double globalstd = std.at<double>(0,0);
		double globalmean = mean.at<double>(0,0);
		cout << "1:"<<globalstd << "" <<"2:"<< globalmean << endl;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				for (int k = 0; k < 3; k++) {
					if (i - 1 >= 0 && i + 1 <= cols - 1 && j - 1 >= 0 && j + 1 <= rows - 1) {
						double localmean = 0;
						double localstd = 0;
						for (int m = 0; m < 3; m++) {
							for (int n = 0; n < 3; n++) {
								localmean += srcImage.at<Vec3b>(i+m-1, j+n-1)[1]/9;
							}
						}
						for (int m = 0; m < 3; m++) {
							for (int n = 0; n < 3; n++) {
								localstd += pow(srcImage.at<Vec3b>(i + m - 1, j + n - 1)[1]-localmean,2) / 9;
							}
						}
						localstd = sqrt(localstd);
						//determine if the pixel should be enhanced
						if (localstd >= k1 * globalstd&&localstd <= k2 * globalstd&&localmean <= k0 * globalmean) {
							dstImage.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(int(srcImage.at<Vec3b>(i, j)[k] * E));
						}
						else
						{
							dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
						}
					}
					else
						dstImage.at<Vec3b>(i, j)[k] = srcImage.at<Vec3b>(i, j)[k];
				}
			}
		}
		break;
	}	
	return dstImage;
};