//
// Created by super on 18-7-16.
//

#include "math_helper.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace cf_tracking
{
    int mod(int dividend, int divisor)
    {
        // http://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
        return ((dividend % divisor) + divisor) % divisor;
    }

    void dftCcs(const cv::Mat& input, cv::Mat& out, int flags)
    {
        cv::dft(input, out, flags);
    }

    void dftNoCcs(const cv::Mat& input, cv::Mat& out, int flags)
    {
        flags = flags | cv::DFT_COMPLEX_OUTPUT;
        cv::dft(input, out, flags);
    }

    // use bi-linear interpolation on zoom, area otherwise
    // similar to mexResize.cpp of DSST
    // http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html
    void depResize(const cv::Mat& source, cv::Mat& dst, const cv::Size& dsize)
    {
        int interpolationType = cv::INTER_AREA;

        if (dsize.width > source.cols
            || dsize.height > source.rows)
            interpolationType = cv::INTER_LINEAR;

        cv::resize(source, dst, dsize, 0, 0, interpolationType);
    }
    // mexResize got different results using different OpenCV, it's not trustable
// I found this bug by running vot2015/tunnel, it happened when frameno+1==22 after frameno+1==21
    void mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method) {
        int interpolation = cv::INTER_LINEAR;

        cv::Size sz = im.size();

        if(!strcmp(method, "antialias")){
            interpolation = cv::INTER_AREA;
        } else if (!strcmp(method, "linear")){
            interpolation = cv::INTER_LINEAR;
        } else if (!strcmp(method, "auto")){
            if(newsz.width > sz.width){ // xxx
                interpolation = cv::INTER_LINEAR;
            }else{
                interpolation = cv::INTER_AREA;
            }
        } else {
            assert(0);
            return;
        }

        resize(im, output, newsz, 0, 0, interpolation);
    }
}
