//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ nada.kth.se                       //
//********************************************************//  

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

#define MATCH_ERROR_THRESHOLD 10
#define FEATURE_MATCH_THRESHOLD 0.01f

void sift_images(const char *im1_name, const char* im2_name,
  double **im1_pts, int *im1_length, double **im2_pts, int *im2_length)
{     
  // Read images using OpenCV
  cv::Mat limg, rimg;

  IplImage *image1, *image2;
  image1 = cvLoadImage(im1_name, 
    CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  image2 = cvLoadImage(im2_name, 
    CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  
  cv::Mat limg_color(image1);
  cv::Mat rimg_color(image2);

  cv::cvtColor(limg_color, limg, CV_BGR2GRAY);
  cv::cvtColor(rimg_color, rimg, CV_BGR2GRAY);

  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
  
  // Perform some initial blurring (if needed)
  cv::GaussianBlur(limg, limg, cv::Size(11,11), 11.0);
  cv::GaussianBlur(rimg, rimg, cv::Size(11,11), 11.0);
        
  cv::imwrite("test/test23_pts.exr", rimg);
  // Initial Cuda images and download images to device
  std::cout << "Initializing data..." << std::endl;
  InitCuda();
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download(); 

  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 0.0f;
  float thresh = FEATURE_MATCH_THRESHOLD;
  InitSiftData(siftData1, 2048, true, true); 
  InitSiftData(siftData2, 2048, true, true);
  ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f);
  ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f);

  // Match Sift features and find a homography
  MatchSiftData(siftData1, siftData2);
  float homography[9];
  int numMatches;
  FindHomography(siftData1, homography, &numMatches, 10000, 0.50f, 1.00f, 5.0);
  int numFit = ImproveHomography(siftData1, homography, 3, 0.80f, 0.95f, 3.0);

  printf("\n");
  printf("\n");
  for (int i = 0; i < 9; ++i)
    printf("%f ", homography[i]);
  printf("\n");
  printf("\n");

  // Print out and store summary data
  GenerateMatchData(siftData1, siftData2, img1, im1_pts, im1_length, im2_pts, im2_length);
  
  std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
  std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numMatches/std::min(siftData1.numPts, siftData2.numPts) << "%" << std::endl;
  cv::imwrite("test/limg_pts.exr", limg);

  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << std::endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++) 
	sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err<100.0f)
	found = true;
      if (err<100.0f || j==sift1[i].match) {
	if (j==sift1[i].match && err<100.0f)
	  std::cout << " *";
	else if (j==sift1[i].match) 
	  std::cout << " -";
	else if (err<100.0f)
	  std::cout << " +";
	else
	  std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of founds: " << numFound << std::endl;
}

void GenerateMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img,
  double **im1_pts, int *im1_length, double **im2_pts, int *im2_length)
{
  int numPts = siftData1.numPts;
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
  float *h_img = img.h_data;

  int numMatches = 0;
  for (int j=0; j<numPts;j++){
    if (sift1[j].match_error< MATCH_ERROR_THRESHOLD)
      ++numMatches;
  }
  *im1_pts = (double *) malloc(sizeof(double) * numMatches * 3);
  *im2_pts = (double *) malloc(sizeof(double) * numMatches * 3);

  int match_num = 0;
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (sift1[j].match_error< MATCH_ERROR_THRESHOLD) {
      (*im1_pts)[3*match_num] = sift1[j].xpos;
      (*im1_pts)[3*match_num + 1] = sift1[j].ypos;
      (*im1_pts)[3*match_num + 2] = j;

      (*im2_pts)[3*match_num] = sift1[k].xpos;
      (*im2_pts)[3*match_num + 1] = sift1[k].ypos;
      (*im2_pts)[3*match_num + 2] = j;

      ++match_num;
    }
  }

  *im2_length = *im1_length = numMatches;
}

int main(void){
  double *im1, *im2;
  int im1l, im2l;

  const char *i1 = "test/lowres_img06.exr";
  const char *i2 = "test/lowres_img23.exr";
  sift_images(i1, i2, &im1, &im1l, &im2, &im2l);

  printf("LENGTH: %d %d\n", im1l, im2l);

  free(im1);
  free(im2);
}
