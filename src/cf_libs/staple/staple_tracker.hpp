//
// Created by super on 18-7-16.
//
#ifndef STAPLE_TRACKER_HPP
#define STAPLE_TRACKER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "scale_estimator.hpp"
#include "cf_tracker.hpp"
#include "staple_debug.hpp"
#include "colour_tracker.hpp"

namespace cf_tracking {
    struct StapleParameters {
        /*************DSST Parameters Start****************/
        //for transfermation
        double padding = static_cast<double>(1);
        double outputSigmaFactor = (1.0 / 16.0); // standard deviation for the desired translation filter output
        double lambda =static_cast<double>(1e-3);// egularization weight
        double learning_rate_cf = 0.01;     // HOG model learning rate
        int templateSize = 150;           // fixed_area = templateSize*templateSize standard size to which we resize the target
        int cellSize = 4;                // hog_cell_size for CF feature

        bool enableTrackingLossDetection = false;
        double psrThreshold = 13.5;
        int psrPeakDel = 1;

        // for scale estimation
        bool enableScaleEstimator = true;
        double scaleSigmaFactor = 1 / 4.0;
        double scaleStep = static_cast<double>(1.02);
        int scaleCellSize = 4;
        int numberOfScales = 33;
        double scaleTemplateArea = 32 * 16;   //Max  template area, in stapel :32*16 = 512
        double learning_rate_scale = static_cast<double>(0.025); //learning rate for scale
        //testing
        bool originalVersion = false;
        int resizeType = cv::INTER_LINEAR;
        bool useFhogTranspose = false;
        /*************DSST Parameters End****************/

        /********Colour feature parameters Start**********/
        bool enableColourTracker = true;
        int n_bins = 2 * 2 * 2 * 2 * 2;             // number of bins for the color histograms (bg and fg models)
        double learning_rate_pwp = static_cast<double>(0.04);    // bg and fg color models learning rate
        double inner_padding = static_cast<double>(0.2);         // defines inner area used to sample colors from the foreground
        /********Colour feature parameters End**********/

        /*********Merge response parameters*********/
        double merge_factor = 0.3;          // fixed interpolation factor - how to linearly combine the two responses
        const char *merge_method = "const_factor";

        double scale_model_factor = 1.0;
        // debugging stuff
        int visualization = 0;              // show output bbox on frame
        int visualization_dbg = 0;          // show also per-pixel scores, desired response and filter output

    };

    class StapleTracker : public CfTracker{
    public:
        /***********************renit/update/updateAt interface added by superlee***************************/
        typedef float T; // set precision here double or float
        static const int CV_TYPE = cv::DataType<T>::type;
        typedef cv::Size_<T> Size;
        typedef cv::Point_<T> Point;
        typedef cv::Rect_<T> Rect;
        typedef DsstFeatureChannels<T> DFC;
        typedef mat_consts::constants<T> consts;

        virtual bool reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual bool reinit(const cv::Mat& image, cv::Rect_<float>& boundingBox);
        virtual bool reinit(const cv::Mat& image, cv::Rect_<double>& boundingBox);
        virtual bool update(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual bool update(const cv::Mat& image, cv::Rect_<float>& boundingBox);
        virtual bool update(const cv::Mat& image, cv::Rect_<double>& boundingBox);
        virtual bool updateAt(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual bool updateAt(const cv::Mat& image, cv::Rect_<float>& boundingBox);
        virtual bool updateAt(const cv::Mat& image, cv::Rect_<double>& boundingBox);
        StapleTracker(StapleParameters paras, StapleDebug<T>* debug = 0);
        ~StapleTracker() {}
        virtual const std::string getId()
        {
            return _ID;
        }
        virtual TrackerDebug* getTrackerDebug()
        {
            return _debug;
        }


    protected:
        bool getTranslationTrainingData(const cv::Mat& image, std::shared_ptr<DFC>& hfNum,
                                        cv::Mat& hfDen, const Point& pos) const;
        bool getTranslationFeatures(const cv::Mat& image, std::shared_ptr<DFC>& features,
                                    const Point& pos, T scale) const;
		// fix some bugs when we update the mode with new boudingbox
		//Add update_at, when we want to update the model with new boudingbox, we set it to true.
        bool updateAtScalePos(const cv::Mat& image, const Point& oldPos, const T oldScale,
                              Rect& boundingBox,const bool update_at = false); 
        bool evalReponse(const cv::Mat &image, const cv::Mat& response,
                         const cv::Point2i& maxResponseIdx,
                         const Rect& tempBoundingBox) const;
        bool detectModel(const cv::Mat& image, cv::Mat& response,
                         cv::Point2i& maxResponseIdx, Point& newPos,
                         T& newScale) const;
        bool updateModel(const cv::Mat& image, const Point& newPos,
                         T newScale);
        void cropFilterResponse(const cv::Mat &response_cf, Size response_size, cv::Mat& output) const ;
        void mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp,
                                           cv::Mat &response, const char *merge_method = "const_factor") const;
    private:
        /***********************the implement of above interface according to different methods*********/
        bool reinit_(const cv::Mat& image, Rect& boundingBox);
        bool update_(const cv::Mat& image, Rect& boundingBox);
        bool updateAt_(const cv::Mat& image, Rect& boundingBox);



    private:
        typedef void(*cvFhogPtr)
                (const cv::Mat& img, std::shared_ptr<DFC>& cvFeatures, int binSize, int fhogChannelsToCopy);
        cvFhogPtr cvFhog = 0;

        typedef void(*dftPtr)
                (const cv::Mat& input, cv::Mat& output, int flags);
        dftPtr calcDft = 0;

        cv::Mat _cosWindow;
        cv::Mat _y;
        std::shared_ptr<DFC> _hfNumerator;
        cv::Mat _hfDenominator;
        cv::Mat _yf;
        Point _pos;
        Size _templateSz;
        Size _baseTargetSz;
        Rect _lastBoundingBox;
        T _scale; // _scale is the scale of the template; not the target
        T _templateScaleFactor; // _templateScaleFactor is used to calc the target scale
        ScaleEstimator<T>* _scaleEstimator = 0;
        Colour_Tracker* _colourTracker = 0;
        int _frameIdx = 1;
        bool _isInitialized;

        const double _MIN_AREA;
        const double _MAX_AREA_FACTOR;
        //const T _INNER_PADDING; // defines inner area used to sample colors from the foreground
        const double _PADDING;
        const T _OUTPUT_SIGMA_FACTOR;
        const T _LAMBDA;
        const T _LEARNING_RATE;
        const T _PSR_THRESHOLD;
        const int _PSR_PEAK_DEL;
        const int _CELL_SIZE;
        const int _TEMPLATE_SIZE;
        const std::string _ID;
        const bool _ENABLE_TRACKING_LOSS_DETECTION;
        const int _RESIZE_TYPE;
        const T _MERGE_FACTOR;
        const bool _ORIGINAL_VERSION;
        const bool _USE_CCS;
        StapleDebug<T>* _debug;
    };
}
#endif