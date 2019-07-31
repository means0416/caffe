#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

double ***Make_matrix()
{
    Mat image;
    double ***p;
    image = imread("4.png", IMREAD_GRAYSCALE);

    if(image.empty())
    {
        cout<<"Could not open or find the image"<<endl;
        return 0;
    }

    p = (double***)malloc(image.channels()*sizeof(double**)); 
    for(int cha=0; cha<image.channels(); cha++){   
        *(p+cha) = (double**)malloc(image.rows*sizeof(double*));
        for(int row = 0; row < image.rows; row++) {
           *(*(p+cha)+row) = (double*)malloc(image.cols*sizeof(double));
           for(int col = 0; col < image.cols; col++ ){
               *(*(*(p+cha)+row)+col) = image.at<uchar>(row,col);
           }
        }
    }
    return p;
}