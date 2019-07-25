#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
#include<time.h>
using namespace cv;
using namespace std;

struct For_value{
    int m;   // 메모리 크기 변수 
    int f;   // filter size 
    int n;  // 3차원 확장을 위한 변수
    int i;   // 2차원 확장을 위한 변수
    int j;   // 1차원 값 할당을 위한 변수 
    int z;   // 행렬 연산을 위한 변수  
    int a;
    int f_i;
    int f_j;     
};
typedef struct For_value For;

int***Make_matrix(int col_s, int row_s, int cha_s);
double **Filter_matrix_p(int n);
double **Filter_matrix(int n);
int ***Padding(int cha_s, int pad, int col_s, int row_s, int ***p);
int***Convolution(int cha_s, int stride, int col_s,  int row_s, int f_size, int p_size, int ***p_data, double **filter);
int ***Maxpooling(int cha_s, int stride, int col_s, int row_s,  int f_size,  int ***p_data, double **filter);
double ***Activation(int cha_s, int col_s, int row_s, int ***input);
double **Filter_Blur_3X3();
double **Filter_Blur_5X5();
double **Filter_Edge_detector();

int main(){

    For v;
    Mat image;
    image = imread("test11.jpg", IMREAD_COLOR);
    if(image.empty())
    {
        cout<<"Could not open or find the image"<<endl;
        return -1;
    }
    namedWindow("Original", WINDOW_AUTOSIZE); //file export, to make window
    imshow("Original", image); //show image, to show the window right before maked file

    int col_s = image.cols;
    int row_s = image.rows;
    int cha_s = image.channels();
   // int scalar = image.Scalar();
    int ***i_data;
    int ***pad_data;
    int ***pool_m_data; 
    int ***result_data;
    double **filter_c_data;
    double **filter_p_data;
    int padding;
    int stride;
    int pad = row_s+2*padding;
    int n;
    int c=0;

    // unsigned char type = image.type();
    // int **filter;
    //filter = (double**)malloc(sizeof(double*)*)


    printf("---------------------------------------\n");
    printf("1. Input data \n");
    printf("Column size : %d, Row size : %d, Channel Qty : %d \n", col_s, row_s, cha_s);
    i_data = Make_matrix(col_s, row_s, cha_s);
    printf("\n\n");
    while( c == 0)
    { 
    printf("2. Filter data \n");
    printf("Choose filter in below \n");
    printf("   1. Blur Filter (3X3) \n");
    printf("   2. Blur Filter (5X5) \n");
    printf("   3. Edge detect Filter \n");
    printf("   4. Make Filter \n");
    printf("   5. Max Pooling \n");
    printf("   User  : ");
    scanf("%d", &n);
    
   
        if( n == 1)
        {   
            printf("***Blur Filter (3X3)***\n");     
            printf("Filter data is in below \n");
            filter_c_data = Filter_Blur_3X3();
            v.f = 3;
            c = 1;
        } 
        else if( n == 2 )
        {
            printf("***Blur Filter (5X5)***\n");    
            printf("Filter data is in below \n");
            filter_c_data = Filter_Blur_5X5();
            v.f = 5;
            c = 1;
        }
        else if( n == 3 )
        {   
            printf("***Edge detect Filter***\n");     
            printf("Filter data is in below \n");
            filter_c_data = Filter_Edge_detector();
            v.f = 3;
            c = 1;
        }
        else if(n == 4 )
        {
            printf("***Make Filter***\n");
            printf("Please determine filter data size : ");
            scanf("%d", &v.f);            
            printf("Filter data is in below \n");
            filter_c_data = Filter_matrix(v.f);
            c = 1;
        } 
        else if(n == 5 )
        {
            printf("***Max Pooling***\n");
            printf("Please determine filter data size : ");
            scanf("%d", &v.f);            
            printf("Filter data is in below \n");
            filter_p_data = Filter_matrix_p(v.f);
            c = 1;
        } 
        else
        {
            printf("ERROR : Please choose again \n\n\n\n");  
        }
    }    
    
    printf("\n\n");
    
    if(n !=5){
    printf("3. Padding \n");
    printf("Please determine padding size : ");
    scanf("%d", &padding);
    printf("Padding result like in below \n");
    pad_data = Padding(cha_s, padding, col_s, row_s, i_data);// v.n : input Qty, v.m : input data, v.f : filter data
    printf("\n\n");
    printf("4. Convolution \n");
    printf("Please determine Stride : ");
    scanf("%d", &stride);

    printf("Convolution result like in below \n");
    result_data = Convolution(cha_s, stride, col_s, row_s, v.f, padding, pad_data, filter_c_data);                
    }
    else
    {
    printf("3. Max Pooling \n");
    printf("Please determine Stride : ");
    scanf("%d", &stride);
    printf("Max Pooling result like in below \n");
    pool_m_data = Maxpooling(cha_s, stride, col_s, row_s, v.f,  i_data, filter_p_data); // v.n : input Qty, v.m : input data, v.f : filter data
    printf("\n\n");
    }
    int con_row_size = (row_s-v.f+2*padding)/stride +1;
    int con_col_size = (col_s-v.f+2*padding)/stride +1;
    int pool_col_size = (col_s-v.f)/stride +1;
    int pool_row_size = (row_s-v.f)/stride +1;

    if(n !=5)
    {
    Mat ConvolutionResult( con_row_size,  con_col_size,  CV_8UC3, Scalar(0, 0, 0)); // CV_8UC3 : 8-bit unsigned integer matrix with 3 channels // U : unsigned (0~255) //Scalar(0,0,0) : color
   // ConvolutionResult.create( con_row_size,  con_col_size,  CV_8UC3); //CV_32SC3 : 32bit color
    for(int z = 0 ; z<cha_s ; z++){
        for(int j = 0 ; j<con_row_size ; j++){
            for(int i = 0 ; i<con_col_size ; i++){
                ConvolutionResult.at<cv::Vec3b>(j,i)[z] = result_data[z][j][i];
            }
        }
    } 

    imwrite("convolution.jpg", ConvolutionResult); // image save ("file name. file type , Mat parameter")    
    namedWindow("convoltion result", WINDOW_AUTOSIZE); // make window ("Window name" , window size)
	imshow("convoltion result", ConvolutionResult); // image show ("Window name", Mat parameter(What users want))
    } 
    else 
    {
    Mat PoolingResult( pool_row_size,  pool_col_size,   CV_8UC3, Scalar(0, 0, 0));
    //PoolingResult.create( pool_row_size,  pool_col_size,  CV_32SC3); //CV_32SC3 : 32bit color
    for(int z = 0 ; z<cha_s ; z++){
        for(int j = 0 ; j<pool_row_size ; j++){
            for(int i = 0 ; i<pool_col_size ; i++){
                    PoolingResult.at<cv::Vec3b>(j,i)[z] = pool_m_data[z][j][i];
            }
        }
    }   
    
    imwrite("Pooling.jpg", PoolingResult); // image save ("file name. file type , Mat parameter")
    namedWindow("Pooling result", WINDOW_AUTOSIZE); // make window ("Window name" , window size)
	imshow("Pooling result", PoolingResult); // image show ("Window name", Mat parameter(What users want))    
    }

    waitKey(0);
    if(n != 5)
    {   
        for(v.z= 0; v.z <cha_s ; v.z++) {
                        for(v.i = 0; v.i < pad ; v.i++){
                            free(*(*(pad_data+v.z)+v.i));
                        } free(*(pad_data+v.z));
        }
        free(pad_data); 

        for(v.z= 0; v.z <cha_s ; v.z++) {
                        for(v.i = 0; v.i < row_s ; v.i++){
                            free(*(*(i_data+v.z)+v.i));
                        } free(*(i_data+v.z));
        }
        free(i_data);

        for(v.i= 0; v.i <v.f ; v.i++) free(*(filter_c_data+v.i));
        free(filter_c_data);

        for(v.z= 0; v.z <cha_s ; v.z++) {
                     for(v.i = 0; v.i < con_row_size ; v.i++){
                            free(*(*(result_data+v.z)+v.i));
                        } free(*(result_data+v.z));
        }
        free(result_data); 
    }
    else
    {    
        for(v.z= 0; v.z <cha_s ; v.z++) {
                        for(v.i = 0; v.i < row_s ; v.i++){
                            free(*(*(i_data+v.z)+v.i));
                        } free(*(i_data+v.z));
        }
        free(i_data);   

        for(v.i= 0; v.i <v.f ; v.i++) free(*(filter_p_data+v.i));
        free(filter_p_data);

        for(v.z= 0; v.z <cha_s ; v.z++) {
                        for(v.i = 0; v.i < pool_row_size ; v.i++){
                            free(*(*(pool_m_data+v.z)+v.i));
                        } free(*(pool_m_data+v.z));
        }
        free(pool_m_data);
    }        
    return 0;
}

double **Filter_Blur_3X3()
{
    For v;
    int fsize=3;
    double **p;
    p = (double**)malloc(fsize*sizeof(double*));
    for(v.i = 0 ; v.i < fsize ; v.i++){
           *(p+v.i)=(double*)malloc(fsize*sizeof(double));
    }

    double F[3][3] = {{1,2,1},
                                 {2,4,2},
                                 {1,2,1}};  
    
    for(v.i = 0; v.i < fsize ; v.i++)
        {
            for(v.j = 0; v.j < fsize ; v.j++)
            {
                *(*(p+v.i)+v.j) = F[v.i][v.j]*(0.0625);
                printf("  %5lf", p[v.i][v.j]);
            }
            printf("\n\n");
        }
        printf("---------------------------------------\n");

    return p;    
}

double **Filter_Blur_5X5()
{
    For v;
    int fsize=5;
    double **p;
    p = (double**)malloc(fsize*sizeof(double*));
    for(v.i = 0 ; v.i < fsize ; v.i++){
           *(p+v.i)=(double*)malloc(fsize*sizeof(double));
    }

    double F[5][5] = {{1,  4,  6,  4,  1},
                                 {4, 16, 24, 16, 4},
                                 {6, 24, 36, 24, 6},
                                 {4, 16, 24, 16, 4},
                                 {1,  4,  6,  4,  1}};  
    
    for(v.i = 0; v.i < fsize ; v.i++)
        {
            for(v.j = 0; v.j < fsize ; v.j++)
            {
                *(*(p+v.i)+v.j) = F[v.i][v.j]*(0.00390625);
                printf("  %5lf", p[v.i][v.j]);
            }
            printf("\n\n");
        }
        printf("---------------------------------------\n");

    return p;       
}

double **Filter_Edge_detector()
{
    For v;
    int fsize=3;
    double **p;
    p = (double**)malloc(fsize*sizeof(double*));
    for(v.i = 0 ; v.i < fsize ; v.i++){
           *(p+v.i)=(double*)malloc(fsize*sizeof(double));
    }

    double F[3][3] = {{-1, -1, -1},
                                 {-1, 8, -1},
                                 {-1, -1, -1}};  
    
    for(v.i = 0; v.i < fsize ; v.i++)
        {
            for(v.j = 0; v.j < fsize ; v.j++)
            {
                *(*(p+v.i)+v.j) = F[v.i][v.j];
                printf("  %5lf", p[v.i][v.j]);
            }
            printf("\n\n");
        }
        printf("---------------------------------------\n");

    return p;      
}

int bubble_sort(int **list, int n)
{
     int m, i, j, z, temp;
    int max;
    for(z=0;z<n;z++){
        for(i=0;i<n;i++){
            for(j=0;j<i;j++){
                if(*(*(list+z)+i)>*(*(list+z)+j))
                {
                    temp = *(*(list+z)+i);
                    *(*(list+z)+i) = *(*(list+z)+j); 
                    *(*(list+z)+j) = temp;   
                }
            }
        }
    }

    for(z=0;z<n;z++)
    {
        for(m=0;m<z;m++)
        {
            if(*list[z]>*list[m])
            {
                temp = *list[z];
                *list[z] = *list[m];
                *list[m] =temp;
            }
        }
    }

   max = *list[0];
       
return max;
}

int ***Maxpooling(int cha_s, int stride, int col_s, int row_s,  int f_size,  int ***p_data, double **filter)
{
    For v;

    int pool_col_size = (col_s-f_size)/stride +1;
    int pool_row_size = (row_s-f_size)/stride +1;
    int ***result;
    int **bubble;
    int max;
    
    bubble = (int**)malloc(f_size*sizeof(int*));
    for(int i = 0; i<f_size ; i++)
    {
        *(bubble+i) = (int*)malloc(f_size*sizeof(int));
    }

    result=(int***)malloc(sizeof(int**)*cha_s);
    for(v.z = 0 ; v.z < cha_s ; v.z++){
        *(result+v.z)=(int**)malloc(sizeof(int*)*pool_row_size);
        for(v.i = 0 ; v.i < pool_row_size ; v.i++){
            *(*(result+v.z)+v.i) = (int*)malloc(sizeof(int)*pool_col_size);
        }
    }

    for(v.z = 0 ; v.z <cha_s ; v.z++){ // Result reset
        for(v.i = 0 ; v.i<pool_row_size ; v.i++){
            for(v.j = 0 ; v.j<pool_col_size ; v.j++){
                result[v.z][v.i][v.j] = 0;
            }
        }
    }

    for(v.z = 0; v.z <cha_s; v.z++){
        for(v.i = 0; v.i<pool_row_size; v.i+=1){
            for(v.j = 0; v.j<pool_col_size; v.j+=1){
                for(v.f_i = 0; v.f_i<f_size; v.f_i++){
                    for(v.f_j= 0;v.f_j<f_size; v.f_j++){
                            bubble[v.f_i][v.f_j] = p_data[v.z][v.f_i+v.i*stride][v.f_j+v.j*stride]*filter[v.f_i][v.f_j];
                    }
                } 
                max = bubble_sort(bubble, f_size);
                result[v.z][v.i][v.j] = max; 
            }  
        }
    }

    for(v.z = 0 ; v.z <cha_s ; v.z++){
        printf("[%d/%d] Max pooling start !!!\n", v.z+1, cha_s);
        for(v.i = 0 ; v.i<pool_row_size ; v.i++){
            for(v.j = 0 ; v.j<pool_col_size ; v.j++){
             //   printf(" %5d", result[v.z][v.i][v.j] );
            }
    //    printf("\n\n");
        }
    printf("[%d/%d] Max pooling finish !!!\n", v.z+1, cha_s);
    printf("---------------------------------------\n");
    }
    for(v.i= 0; v.i <f_size ; v.i++) free(*(bubble+v.i));
    free(bubble);  
    return result;
}

int ***Convolution(int cha_s, int stride, int col_s,  int row_s, int f_size, int p_size, int ***p_data, double **filter)
{   
    For v;
    int con_row_size = (row_s-f_size+2*p_size)/stride +1;
    int con_col_size = (col_s-f_size+2*p_size)/stride +1;
    int ***result;

    result=(int***)malloc(sizeof(int**)*cha_s);
    for(v.z = 0 ; v.z < cha_s ; v.z++){
        *(result+v.z)=(int**)malloc(sizeof(int*)*con_row_size);
        for(v.i = 0 ; v.i < con_row_size ; v.i++){
            *(*(result+v.z)+v.i) = (int*)malloc(sizeof(int)*con_col_size);
        }
    }

    for(v.z = 0 ; v.z <cha_s ; v.z++){ // Result reset
        for(v.i = 0 ; v.i<con_row_size ; v.i++){
            for(v.j = 0 ; v.j<con_col_size ; v.j++){
                result[v.z][v.i][v.j] = 0;
            }
        }
    }
    double time[3];
    for(v.z = 0; v.z <cha_s; v.z++){
        clock_t begin = clock(); // convolution time check!!
        for(v.i = 0; v.i<con_row_size; v.i+=1){
            for(v.j = 0; v.j<con_col_size; v.j+=1){
                for(v.f_i = 0; v.f_i<f_size; v.f_i++){
                    for(v.f_j= 0;v.f_j<f_size; v.f_j++){
                            result[v.z][v.i][v.j] += p_data[v.z][v.f_i+v.i*stride][v.f_j+v.j*stride]*filter[v.f_i][v.f_j];
                    }    
                }  
                if(result[v.z][v.i][v.j] > 255)
                {
                    result[v.z][v.i][v.j] = 255;
                }
                else if(result[v.z][v.i][v.j]<0)
                { 
                    result[v.z][v.i][v.j]=0;
                }   
            }   
        }
       clock_t end = clock();
       time[v.z] = double(end-begin);
    }
    
    for(v.z = 0 ; v.z <cha_s ; v.z++){
        printf("[%d/%d] Convolution start !!!\n", v.z+1, cha_s);
        for(v.i = 0 ; v.i<con_row_size ; v.i++){
            for(v.j = 0 ; v.j<con_col_size ; v.j++){
                //printf(" %5d", result[v.z][v.i][v.j] );
            }
        //printf("\n\n");
        }
    printf("[%d/%d] Convolution finish !!!\n", v.z+1, cha_s);   
    printf("---------------------------------------\n");
   printf("Convolution sec : %.2lf sec \n", time[v.z]/CLOCKS_PER_SEC); //CLOCKS_PER_SEC : to check unit second
    }   
  return result;
}

int ***Padding(int cha_s, int pad, int col_s, int row_s, int ***p ) // d : input data Qty , n : input data size
{
    For v;
    int ***p_m;
    int ds_col = col_s+(2*pad); //data size
    int ds_row = row_s+(2*pad); //data size

    p_m = (int***)malloc(cha_s*sizeof(int**));
    for(v.z=0; v.z<cha_s; v.z++)
    {   
        *(p_m+v.z) = (int**)malloc(ds_row*sizeof(int*));
        for(v.i = 0; v.i < ds_row ; v.i++) 
         {
           *(*(p_m+v.z)+v.i) = (int*)malloc(ds_col*sizeof(int));
        }
    }
   
    for(v.z = 0 ; v.z<cha_s ; v.z++){
        for(v.i = 0 ; v.i<ds_row ; v.i++){
   
            for(v.j =0 ; v.j<ds_col ; v.j++){

                if ((pad<=v.i)&&(v.i<row_s+pad)&&(pad<=v.j)&&(v.j<col_s+pad)){
                    p_m[v.z][v.i][v.j] = p[v.z][v.i-pad][v.j-pad];  
                      //  *(*(*(p_m+v.z)+v.i)+v.j)   = *(*(*(p+v.z)+v.i-pad)+v.j-pad); 
                }else  {
                    p_m[v.z][v.i][v.j] = 0;
                    //*(*(*(p_m+v.z)+v.i)+v.j)  = 0; 
                }
            }   
        }
    }

 /*    
    for(v.z = 0; v.z<cha_s ; v.z++){
        for(v.i = 0; v.i<ds_row;v.i++){
            for(v.j =0; v.j<ds_col;v.j++){
                    printf("% 5d", p_m[v.z][v.i][v.j]);                             
            }
        printf("\n\n");
        }
    printf("---------------------------------------\n");
    }
 */   
    return p_m;
}

double **Filter_matrix_p(int n) // d : input data Qty , n : input data size
{
    For v;

    double **p;
    p = (double**)malloc(n*sizeof(double*));
    for(v.i = 0 ; v.i < n ; v.i++){
           *(p+v.i)=(double*)malloc(n*sizeof(double));
    }

    for(v.i = 0 ; v.i <n ; v.i++){
            for(v.j = 0; v.j < n ; v.j++)
            {
                *(*(p+v.i)+v.j) = 1;
            }
    }
       
    
    printf("  Filter matrix for pooling is like in below \n");
    

    for(v.i = 0; v.i < n ; v.i++)
        {
            for(v.j = 0; v.j < n ; v.j++)
            {
                printf("  %5lf", p[v.i][v.j]);
            }
            printf("\n\n");
        }
        printf("---------------------------------------\n");

    return p;
}


// CH Qyt modification
double **Filter_matrix(int n) // d : input data Qty , n : input data size 
{
    For v;
    double **p;
    p = (double**)malloc(n*sizeof(double*));
    for(v.i = 0 ; v.i < n ; v.i++){
           *(p+v.i)=(double*)malloc(n*sizeof(double));
    }

    for(v.i = 0 ; v.i <n ; v.i++){
            printf("   현재는 %d 행 입력 구간 입니다 \n", v.i+1);
            for(v.j = 0; v.j < n ; v.j++)
            {
                printf("   원하는 메모리 행렬 값을 입력해 주세요 [%d열] :  ", v.j+1);
                scanf("%lf",*(p+v.i)+v.j);
            }
    }
       
    
    printf("   입력하신 값은 아래와 같습니다 \n\n");
    

    for(v.i = 0; v.i < n ; v.i++)
        {
            for(v.j = 0; v.j < n ; v.j++)
            {
                printf("  %5lf", p[v.i][v.j]);
            }
            printf("\n\n");
        }
        printf("---------------------------------------\n");

    return p;
}

int ***Make_matrix(int col_s, int row_s, int cha_s)
{
    For v;
    Mat image;
    int  z, i,  j;
    int ***p;
    image = imread("test11.jpg", IMREAD_COLOR);

    p = (int***)malloc(cha_s*sizeof(int**)); 
    for(v.z=0; v.z<cha_s; v.z++){   
        *(p+v.z) = (int**)malloc(row_s*sizeof(int*));
        for(v.i = 0; v.i < row_s ; v.i++) {
           *(*(p+v.z)+v.i) = (int*)malloc(col_s*sizeof(int));
        }
    }

    for(v.z=0; v.z<cha_s; v.z++)
    {   
        for(v.i = 0; v.i < row_s ; v.i++) 
        {   
            for(v.j = 0; v.j < col_s ; v.j++)
            {               
                p[v.z][v.i][v.j] = image.at<cv::Vec3b>(v.i,v.j)[v.z] ;   //*(*(*(p+v.z)+v.i)+v.j)
            }
        }
    }
/*
    printf("   Print Data is in below \n\n");

    for(v.z=0; v.z<cha_s; v.z++)
    {   
        printf(" %d.현재는 [%d/%d]번 구간 입니다 \n", v.z+1, v.z+1, cha_s);
        for(v.i = 0; v.i < row_s ; v.i++)
        {
            for(v.j = 0; v.j < col_s ; v.j++)
            {
               printf("  %5d", *(*(*(p+v.z)+v.i)+v.j));
            }
            printf("\n\n");
        }
        printf("---------------------------------------\n");
    
    }
*/
 

    //for(v.i = 0; v.i < v.n ; v.i++) free(*(p+v.i)); // n차원 메모리 할당 종료
    //free(p);    // 1차원 메모리 할당 종료
    return p;
}

double ***Activation(int cha_s, int col_s, int row_s, int ***input){
    For v;
    double ***result;

    input = (int***)malloc(cha_s*sizeof(int**)); 
    for(v.z=0; v.z<cha_s; v.z++){   
        *(input+v.z) = (int**)malloc(row_s*sizeof(int*));
        for(v.i = 0; v.i < row_s ; v.i++) {
           *(*(input+v.z)+v.i) = (int*)malloc(col_s*sizeof(int));
        }
    }

    for(v.z=0; v.z<cha_s; v.z++)
    {   
        for(v.i = 0; v.i < row_s ; v.i++) 
        {   
            for(v.j = 0; v.j < col_s ; v.j++)
            {               
                result[v.z][v.i][v.j] = 1/(1+exp(input[v.z][v.i][v.j]));   //*(*(*(p+v.z)+v.i)+v.j)
            }
        }
    }
    return result;
}