#ifndef __wave_VISION_FAST_HPP__
#define __wave_VISION_FAST_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "wave/utils/utils.hpp"


namespace wave {

class FastDetector
{
public:
    bool configured;

    int threshold;
    bool nonmax_suppression;

    FastDetector(void);
    int configure(int threshold, bool nonmax_suppression);
    int detect(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints);
    int detect(cv::Mat &image, std::vector<cv::Point2f> &points);
    int detect(cv::Mat &image, MatX &points);
};

}  // end of wave namespace
#endif