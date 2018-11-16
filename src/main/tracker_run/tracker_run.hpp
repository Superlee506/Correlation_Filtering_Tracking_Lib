//
// Created by super on 18-7-16.
//
/*
// Original file: https://github.com/Itseez/opencv_contrib/blob/292b8fa6aa403fb7ad6d2afadf4484e39d8ca2f1/modules/tracking/samples/tracker.cpp
// * Refactor file: Move target selection to separate class/file
// * Replace command line argumnets
// * Change tracker calling code
// * Add a variety of additional features
*/

#ifndef TRACKER_RUN_HPP_
#define TRACKER_RUN_HPP_

#include <tclap/CmdLine.h>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "cf_tracker.hpp"
#include "tracker_debug.hpp"
#include "dsst_tracker.hpp"
#include "image_acquisition.hpp"

struct Parameters{
    std::string sequencePath;
    std::string outputFilePath;
    std::string imgExportPath;
    std::string expansion;
    cv::Rect_<double> initBb;
    int device;
    int startFrame;
    int endFrame;
    bool showOutput;
    bool paused;
    bool repeat;
    bool isMockSequence;
};

class TrackerRun
{
public:
    TrackerRun(std::string windowTitle);
    virtual ~TrackerRun();
    bool start(int argc, const char** argv);
    void setTrackerDebug(cf_tracking::TrackerDebug* debug);

private:
    Parameters parseCmdArgs(int argc, const char** argv);
    bool init();
    bool run();
    bool update();
    void printResults(const cv::Rect_<double>& boundingBox, bool isConfident, double fps);

protected:
    virtual cf_tracking::CfTracker* parseTrackerParas(TCLAP::CmdLine& cmd, int argc, const char** argv) = 0;
private:
    cv::Mat _image;
    cf_tracking::CfTracker* _tracker;
    std::string _windowTitle;
    Parameters _paras;
    cv::Rect_<double> _boundingBox;
    ImageAcquisition _cap;
    std::ofstream _resultsFile;
    TCLAP::CmdLine _cmd;
    cf_tracking::TrackerDebug* _debug;
    int _frameIdx;
    bool _isPaused = false;
    bool _isStep = false;
    bool _exit = false;
    bool _hasInitBox = false;
    bool _isTrackerInitialzed = false;
    bool _targetOnFrame = false;
    bool _updateAtPos = false;
};

#endif
