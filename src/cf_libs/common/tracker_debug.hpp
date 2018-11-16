//
// Created by super on 18-7-16.
//
#ifndef TRACKER_DEBUG_HPP_
#define TRACKER_DEBUG_HPP_

namespace cf_tracking
{
    class TrackerDebug
    {
    public:
        virtual ~TrackerDebug(){}

        virtual void init(std::string outputFilePath) = 0;
        virtual void printOnImage(cv::Mat& image) = 0;
        virtual void printConsoleOutput() = 0;
        virtual void printToFile() = 0;
    };
}

#endif
