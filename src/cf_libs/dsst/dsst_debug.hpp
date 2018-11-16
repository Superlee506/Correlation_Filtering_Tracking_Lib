
#ifndef DSST_DEBUG_HPP_
#define DSST_DEBUG_HPP_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>

#include "tracker_debug.hpp"

namespace cf_tracking
{
    template<typename T>
    class DsstDebug : public TrackerDebug
    {
    public:
        DsstDebug() :
            _maxResponse(0),
            _psrClamped(0),
            _targetSizeArea(0)
        {}

        virtual ~DsstDebug()
        {
            if (_outputFile.is_open())
                _outputFile.close();
        }

        virtual void init(std::string outputFilePath)
        {
            namedWindow(_SUB_WINDOW_TITLE, cv::WINDOW_NORMAL);
            namedWindow(_RESPONSE_TITLE, cv::WINDOW_NORMAL);
            _outputFile.open(outputFilePath.c_str());
        }

        virtual void printOnImage(cv::Mat& image)
        {
            _ss.str("");
            _ss.clear();
            _ss << "Max Response: " << _maxResponse;
            putText(image, _ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

            _ss.str("");
            _ss.clear();
            _ss << "PSR Clamped: " << _psrClamped;
            putText(image, _ss.str(), cv::Point(20, 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

            _ss.str("");
            _ss.clear();
            _ss << "Area: " << _targetSizeArea;
            putText(image, _ss.str(), cv::Point(image.cols - 100, 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
        }

        virtual void printConsoleOutput()
        {
        }

        virtual void printToFile()
        {
            _outputFile << _maxResponse << "," << _psrClamped << std::endl;
        }

        void showPatch(const cv::Mat& patchResized)
        {
            imshow(_SUB_WINDOW_TITLE, patchResized);
        }

        void setPsr(double psrClamped)
        {
            _psrClamped = psrClamped;
        }

        void showResponse(const cv::Mat& response, double maxResponse)
        {
            cv::Mat responseOutput = response.clone();
            _maxResponse = maxResponse;
            imshow(_RESPONSE_TITLE, responseOutput);
        }

        void setTargetSizeArea(T targetSizeArea)
        {
            _targetSizeArea = targetSizeArea;
        }

    private:
        const std::string _SUB_WINDOW_TITLE = "Sub Window";
        const std::string _RESPONSE_TITLE = "Response";
        double _maxResponse;
        double _psrClamped;
        T _targetSizeArea;
        std::stringstream _ss;
        std::ofstream _outputFile;
    };
}

#endif
