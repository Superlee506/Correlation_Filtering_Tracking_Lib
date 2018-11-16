//
// Created by super on 18-7-20.
//

#ifndef CFTRACKING_COLOUR_TRACKER_HPP
#define CFTRACKING_COLOUR_TRACKER_HPP
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "math_helper.hpp"
namespace cf_tracking {
    template<typename T>
    struct ColourParameters {
        int n_bins = 2 * 2 * 2 * 2 * 2;// number of bins for the color histograms (bg and fg models)
        T learningRate = static_cast<T>(0.04);// bg and fg color models learning rate
        T inner_padding = static_cast<T>(0.2);// defines inner area used to sample colors from the foreground
        bool debugOutput = false;//For debug
        int fixed_area = 150*150;
        int cell_size = 4; // is used to determine the size of response
        T padding = 1;
        bool originalVersion = false;
        int resizeType = cv::INTER_LINEAR;

    };

//    template<typename T>
    class Colour_Tracker{
    public:
        typedef float T; // set precision here double or float
        typedef cv::Size_<T> Size;
        typedef cv::Point_<T> Point;
        typedef cv::Rect_<T> Rect;
        typedef mat_consts::constants<T> consts;
        Colour_Tracker(ColourParameters<T> paras);
        ~Colour_Tracker() {}

        bool reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        bool reinit(const cv::Mat& image, cv::Rect_<float>& boundingBox);
        bool reinit(const cv::Mat& image, cv::Rect_<double>& boundingBox);

        bool updateColourModel(const cv::Mat& image, const Point& pos,
                               const T& currentScaleFactor);
        bool calColourResponse(const cv::Mat& image, cv::Mat& colourResponse);
        Size getNormDeltaArea(){
            return _norm_delta_area;
        }
        T getAreaResizeScale(){
            return _area_resize_scale;
        }
        Size getBgArea(){
            return _bg_area;
        }
        Size getNormalBgArea(){
            return _norm_bg_area;
        }
        Size getNormalTargetSz(){
            return _norm_target_sz;
        }
        Size getFgArea(){
            return _fg_area;
        }
        T getTmpResizeScale(){
            return _tmp_resize_scale;
        }
        Size getNoramlTargetSize(){
            return _norm_target_sz;
        }

    protected:

        // UPDATEHISTMODEL create new models for foreground and background or update the current ones
        void updateHistModel_(bool is_new_model, cv::Mat &patch, T learning_rate_pwp=0);
        void getColourMap_(const cv::Mat &patch, cv::Mat& output);
        void getCenterLikelihood_(const cv::Mat &object_likelihood, const Size m, cv::Mat& center_likelihood);

    private:
        /***********************the implement of above interface according to different methods*********/
        bool reinit_(const cv::Mat& image, Rect& boundingBox);
        void reinitBgFgArea_(const cv::Mat& image, const T currentScaleFactor);

    private:
        cv::MatND _bg_hist;
        cv::MatND _fg_hist;
        bool _grayscale_sequence;
        int _frameIdx;
        bool _isInitialized;
        Point _pos;

        Size _bg_area;
        Size _fg_area;
        Size _target_sz;
        Size _base_target_sz;
        T _area_resize_scale;
        T _tmp_resize_scale;
        Size _response_size;

        Size _norm_bg_area;
        Size _norm_target_sz;
        Size _norm_delta_area;
        Size _norm_pwp_search_area;

        const int _N_BINS;
        const T _LEARNING_RATE;
        const T _INNER_PADDING;
        const bool _DEBUG_OUTPUT;
        const int _FIXED_AREA;
        const int _CELL_SIZE;
        const T _PADDING;
        const bool _ORIGINAL_VERSION;
        const int _RESIZE_TYPE;

    };
}

#endif //CFTRACKING_COLOUR_TRACKER_HPP
