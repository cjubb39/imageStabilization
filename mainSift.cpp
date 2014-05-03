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

#include <stdio.h>

#include "cudaImage.h"
#include "cudaSift.h"
#include "mainSift.h"

#define MATCH_ERROR_THRESHOLD 10
#define FEATURE_MATCH_THRESHOLD 0.01f

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
double ComputeSingular(CudaImage *img, CudaImage *svd);
void GenerateMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img, double **, int *, double **, int *);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);


void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img)
{
  int numPts = siftData1.numPts;
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (sift1[j].match_error<10) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
      std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
      std::cout << "error=" << (int)sift1[j].match_error << "  ";
      std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
      std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ")" << std::endl;
      if (0) std::cout << "  delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
#endif
#if 0
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
  int x = (int)(sift1[j].xpos + dx*l/len);
  int y = (int)(sift1[j].ypos + dy*l/len);
  h_img[y*w+x] = 255.0f;
      } 
#endif
    }
#if 1
    int x = (int)(sift1[j].xpos+0.5);
    int y = (int)(sift1[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
#endif
  }
  std::cout << std::setprecision(6);
}


void sift_images(const char *im1_name, const char* im2_name,
  double **im1_pts, int *im1_length, double **im2_pts, int *im2_length, float* homography)
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
  //float homography[9];
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
  PrintMatchData(siftData1, siftData2, img1);
  
  std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
  std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numMatches/std::min(siftData1.numPts, siftData2.numPts) << "%" << std::endl;
  cv::imwrite("sift_match.exr", limg);

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

fprintf(stderr, "MY NUM MATCHES: %d\n", numMatches);

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
