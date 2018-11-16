//
// Created by super on 18-7-20.
//

#include "colour_tracker.hpp"
using namespace cf_tracking;
using namespace std;
Colour_Tracker::Colour_Tracker(ColourParameters<T> paras) :
        _frameIdx(0),
        _isInitialized(false),
        _N_BINS(paras.n_bins),
        _LEARNING_RATE(paras.learningRate),
        _INNER_PADDING(paras.inner_padding),
        _DEBUG_OUTPUT(paras.debugOutput),
        _FIXED_AREA(paras.fixed_area),
        _CELL_SIZE(paras.cell_size),
        _PADDING(paras.padding),
        _ORIGINAL_VERSION(paras.originalVersion),
        _RESIZE_TYPE(paras.resizeType)
{
    if (_DEBUG_OUTPUT)
    {
        if (CV_MAJOR_VERSION < 3)
        {
            std::cout << "ScaleEstimator: Using OpenCV Version: " << CV_MAJOR_VERSION << std::endl;
            std::cout << "For more speed use 3.0 or higher!" << std::endl;
        }
    }
}

/******************The implement of reinit/update/updateAt added by superlee**********************/
bool Colour_Tracker::reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    return reinit_(image, bb);
}

bool Colour_Tracker::reinit(const cv::Mat& image, cv::Rect_<float>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    return reinit_(image, bb);
}

bool Colour_Tracker::reinit(const cv::Mat& image, cv::Rect_<double>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    return reinit_(image, bb);
}

//bool Colour_Tracker::_reinit(const cv::Mat& image, const Point& pos,
//            const Size& targetSize) {
bool Colour_Tracker::reinit_(const cv::Mat& image, Rect& boundingBox)
{
    _pos.x = boundingBox.x + boundingBox.width * consts::c0_5;
    _pos.y = boundingBox.y + boundingBox.height * consts::c0_5;
    Size targetSize;
    targetSize.width = round(boundingBox.width);
    targetSize.height = round(boundingBox.height);
    _grayscale_sequence = false;
    int n = image.channels();
    if(n==1)
    {
        _grayscale_sequence = true;
    }

    _base_target_sz = targetSize;
    _target_sz.width = round(_base_target_sz.width);
    _target_sz.height = round(_base_target_sz.height);

    _tmp_resize_scale = 1.0;

    //initial _target_sz, _bg_area, fg_are
    reinitBgFgArea_(image, 1);

    _norm_bg_area.width = round(_bg_area.width / _area_resize_scale);
    _norm_bg_area.height = round(_bg_area.height / _area_resize_scale);
    if(_DEBUG_OUTPUT)
    {
        std::cout << "area_resize_factor " << _area_resize_scale << " norm_bg_area.width " <<
                  _norm_bg_area.width << " norm_bg_area.height " << _norm_bg_area.height << std::endl;
    }


    // Correlation Filter (HOG) feature space
    // It smaller that the norm bg area if HOG cell size is > 1
    _response_size.width = floor(_norm_bg_area.width / _CELL_SIZE);
    _response_size.height = floor(_norm_bg_area.height / _CELL_SIZE);

    // given the norm BG area, which is the corresponding target w and h?
    // T norm_target_sz_w = 0.75*_norm_bg_area.width - 0.25*_norm_bg_area.height;
    // T norm_target_sz_h = 0.75*_norm_bg_area.height - 0.25*_norm_bg_area.width;

    T norm_target_sz_w = _target_sz.width * _norm_bg_area.width / _bg_area.width;
    T norm_target_sz_h = _target_sz.height * _norm_bg_area.height / _bg_area.height;
    _norm_target_sz.width = round(norm_target_sz_w);
    _norm_target_sz.height = round(norm_target_sz_h);
    if(_DEBUG_OUTPUT)
    {
        std::cout << "norm_target_sz.width " << _norm_target_sz.width <<
                  " norm_target_sz.height " << _norm_target_sz.height << std::endl;
    }


    // distance (on one side) between target and bg area
    cv::Size norm_pad;

    norm_pad.width = floor((_norm_bg_area.width - _norm_target_sz.width) / 2.0);
    norm_pad.height = floor((_norm_bg_area.height - _norm_target_sz.height) / 2.0);
    //如果target 太大，则pad有可能为0
    norm_pad.width = std::max(int(norm_pad.width),int(0.25*_norm_target_sz.width));
    norm_pad.height = std::max(int(norm_pad.height),int(0.25*_norm_target_sz.height));

    int radius = floor(fmin(norm_pad.width, norm_pad.height));

    // norm_delta_area is the number of rectangles that are considered.
    // it is the "sampling space" and the dimension of the final merged resposne
    // it is squared to not privilege any particular direction
    _norm_delta_area = Size((2*radius+1), (2*radius+1));

    // Rectangle in which the integral images are computed.
    // Grid of rectangles ( each of size norm_target_sz) has size norm_delta_area.
    _norm_pwp_search_area.width = _norm_target_sz.width + _norm_delta_area.width - 1;
    _norm_pwp_search_area.height = _norm_target_sz.height + _norm_delta_area.height - 1;
    if(_DEBUG_OUTPUT)
    {
        std::cout << "norm_pwp_search_area.width " << _norm_pwp_search_area.width <<
                  " norm_pwp_search_area.height " << _norm_pwp_search_area.height<<std::endl;
    }

    cv::Mat patch;

    if (getSubWindow(image, patch, _bg_area, _pos) == false)
        return false;

//    if (_ORIGINAL_VERSION)
//        depResize(patch, patch, _norm_bg_area);
//    else
//        resize(patch, patch, _norm_bg_area, 0, 0, _RESIZE_TYPE);
    // imresize(subWindow, output, model_sz, 'bilinear', 'AntiAliasing', false)
    mexResize(patch, patch, _norm_bg_area, "auto");



    // initialize hist model
    updateHistModel_(true, patch);

}


void Colour_Tracker::reinitBgFgArea_(const cv::Mat& image, const T currentScaleFactor) {


    _target_sz.width = round(_base_target_sz.width*currentScaleFactor);
    _target_sz.height = round(_base_target_sz.height*currentScaleFactor);
    T avg_dim = (_target_sz.width + _target_sz.height) / 2.0;
    _bg_area = Size(floor(_target_sz.width * (1 + _PADDING)),
                    floor(_target_sz.height * (1 + _PADDING)));
//    _bg_area.width = round(_target_sz.width + avg_dim);
//    _bg_area.height = round(_target_sz.height + avg_dim);


    // pick a "safe" region smaller than bbox to avoid mislabeling
//    _fg_area.width = round(_target_sz.width - avg_dim * _INNER_PADDING);
//    _fg_area.height = round(_target_sz.height - avg_dim * _INNER_PADDING);
    _fg_area.width = round(_target_sz.width*(1-_INNER_PADDING));
    _fg_area.height = round(_target_sz.height*(1-_INNER_PADDING));

    // saturate to image size
    Size imsize = image.size();
    //TODO thinks about the bbox of bg
    _bg_area.width = std::min(_bg_area.width, imsize.width - 1);
    _bg_area.height = std::min(_bg_area.height, imsize.height - 1);

    // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
    _bg_area.width = _bg_area.width - int(_bg_area.width - _target_sz.width) % 2;
    _bg_area.height = _bg_area.height - int(_bg_area.height - _target_sz.height) % 2;

    _fg_area.width = _fg_area.width + int(_bg_area.width - _fg_area.width) % 2;
    _fg_area.height = _fg_area.height + int(_bg_area.height - _fg_area.height) % 2;
    if(_DEBUG_OUTPUT)
    {
        std::cout << "bg_area.width " << _bg_area.width << " bg_area.height " <<
                  _bg_area.height << std::endl;
        std::cout << "fg_area.width " << _fg_area.width << " fg_area.height " <<
                  _fg_area.height << std::endl;
    }
    if((currentScaleFactor ==1.0)&&(!_ORIGINAL_VERSION)){
        _tmp_resize_scale = sqrt(T(_bg_area.width*_bg_area.height)/_FIXED_AREA);
    }
    _area_resize_scale = _tmp_resize_scale*currentScaleFactor;


}

bool Colour_Tracker::updateColourModel(const cv::Mat& image,const Point& pos,
                                       const T& currentScaleFactor)
{
    cv::Mat patch;
    //update pos
    _pos = pos;
    //update targe_sz, _bg_area,_fg_area
    reinitBgFgArea_(image,currentScaleFactor);

    if (getSubWindow(image, patch, _bg_area, pos) == false)
        return false;

//    if (_ORIGINAL_VERSION)
//        depResize(patch, patch, _norm_bg_area);
//    else
//        resize(patch, patch, _norm_bg_area, 0, 0, _RESIZE_TYPE);
    mexResize(patch,patch,_norm_bg_area,"auto");
//    getSubwindow(image, _pos, _norm_bg_area, _bg_area, patch);
    // initialize hist model
    updateHistModel_(false, patch,_LEARNING_RATE);
    return true;
}

bool Colour_Tracker::calColourResponse(const cv::Mat &image, cv::Mat &colourResponse) {
    Size pwp_search_area;
    cv::Mat im_patch_pwp;

    pwp_search_area.width = round(_norm_pwp_search_area.width * _area_resize_scale);
    pwp_search_area.height = round(_norm_pwp_search_area.height * _area_resize_scale);

    if (getSubWindow(image, im_patch_pwp, pwp_search_area, _pos) == false)
        return false;
//    if (_ORIGINAL_VERSION)
//        depResize(im_patch_pwp, im_patch_pwp, _norm_pwp_search_area);
//    else
//        resize(im_patch_pwp, im_patch_pwp, _norm_pwp_search_area, 0, 0, _RESIZE_TYPE);
    mexResize(im_patch_pwp,im_patch_pwp,_norm_pwp_search_area,"auto");


    cv::Mat likelihood_map;
    getColourMap_(im_patch_pwp, likelihood_map);

    // each pixel of response_pwp loosely represents the likelihood that
    // the target (of size norm_target_sz) is centred on it
    //return colourResponse
    getCenterLikelihood_(likelihood_map, _norm_target_sz, colourResponse);
//    std::cout<<"colourResponse:"<<std::endl<<colourResponse<<std::endl;
    return true;

}

// GETCOLOURMAP computes pixel-wise probabilities (PwP) given PATCH and models BG_HIST and FG_HIST
void Colour_Tracker::getColourMap_(const cv::Mat &patch, cv::Mat& output)
{
    // check whether the patch has 3 channels
    cv::Size sz = patch.size();
    int h = sz.height;
    int w = sz.width;
    //int d = patch.channels();
    // figure out which bin each pixel falls into
    int bin_width = 256 / _N_BINS;

    // convert image to d channels array
    //patch_array = reshape(double(patch), w*h, d);
    float probg;
    float profg;
    float *P_O = new float[w*h];
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            if (!_grayscale_sequence) {
                cv::Vec3b p = patch.at<cv::Vec3b>(j,i);

                int b1 = floor(p[0] / bin_width);
                int b2 = floor(p[1] / bin_width);
                int b3 = floor(p[2] / bin_width);

                float* histd;

                histd = (float*)_bg_hist.data;
                probg = histd[b1*_N_BINS*_N_BINS + b2*_N_BINS + b3];

                histd = (float*)_fg_hist.data;
                profg = histd[b1*_N_BINS*_N_BINS + b2*_N_BINS + b3];
                P_O[j*w+i] = profg / (profg + probg);

                isnan(P_O[j*w+i]) && (P_O[j*w+i] = 0.0);

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            } else {
                int b = patch.at<uchar>(j,i);

                float* histd;

                histd = (float*)_bg_hist.data;
                probg = histd[b];

                histd = (float*)_fg_hist.data;
                profg = histd[b];

                // xxx
                P_O[j*w+i] = profg / (profg + probg);

                isnan(P_O[j*w+i]) && (P_O[j*w+i] = 0.0);

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            }

        }
    output = cv::Mat(h, w, CV_32FC1, P_O).clone();

    delete[] P_O;
}

// GETCENTERLIKELIHOOD computes the sum over rectangles of size M.
void Colour_Tracker::getCenterLikelihood_(const cv::Mat &object_likelihood, Size m, cv::Mat& center_likelihood)
{
    // CENTER_LIKELIHOOD is the 'colour response'
    cv::Size sz = object_likelihood.size();
    int h = sz.height;
    int w = sz.width;
    int n1 = w - m.width + 1;
    int n2 = h - m.height + 1;
    int area = m.width * m.height;

    cv::Mat temp;
    // integral images
    cv::integral(object_likelihood, temp,CV_32F);

    T *CENTER_LIKELIHOOD = new T[n1*n2];

//    std::cout<<"Type:"<<temp.type()<<std::endl;
//    std::cout<<"Type:"<<object_likelihood.type()<<std::endl;

    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n2; j++) {
            CENTER_LIKELIHOOD[j*n1 + i]
                    = (temp.at<T>(j, i) + temp.at<T>(j+m.height, i+m.width) - temp.at<T>(j, i+m.width) - temp.at<T>(j+m.height, i)) / area;
        }

    // SAT = integralImage(object_likelihood);
    // i = 1:n1;
    // j = 1:n2;
    // center_likelihood = (SAT(i,j) + SAT(i+m(1), j+m(2)) - SAT(i+m(1), j) - SAT(i, j+m(2))) / prod(m);

    center_likelihood = cv::Mat(n2, n1, CV_32FC1, CENTER_LIKELIHOOD).clone();
    delete[] CENTER_LIKELIHOOD;
}

void Colour_Tracker::updateHistModel_(bool is_new_model, cv::Mat &patch, T learning_rate_pwp) {
    // Get BG (frame around target_sz) and FG masks (inner portion of target_sz)

    ////////////////////////////////////////////////////////////////////////
    Size pad_offset1;

    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset1.width = (_bg_area.width - _target_sz.width) / 2;
    pad_offset1.height = (_bg_area.height - _target_sz.height) / 2;

    // difference between bg_area and target_sz has to be even
    if (
            (
                    (pad_offset1.width == round(pad_offset1.width)) &&
                    (pad_offset1.height != round(pad_offset1.height))
            ) ||
            (
                    (pad_offset1.width != round(pad_offset1.width)) &&
                    (pad_offset1.height == round(pad_offset1.height))
            ))
    {
        std::cout<<"difference between bg_area and target_sz has to be even"<<std::endl;
        std::cout<<"pad_offset1.width:"<<pad_offset1.width<<", "
                                                            <<"pad_offset1.height:"<<pad_offset1.height<<std::endl;
        assert(0);
    }

    pad_offset1.width = fmax(pad_offset1.width, 1);
    pad_offset1.height = fmax(pad_offset1.height, 1);

    if (_DEBUG_OUTPUT) {
        std::cout << "pad_offset1 " << pad_offset1 << std::endl;
    }


    cv::Mat bg_mask(_bg_area, CV_8UC1, cv::Scalar(1)); // init bg_mask

    // xxx: bg_mask(pad_offset1(1)+1:end-pad_offset1(1), pad_offset1(2)+1:end-pad_offset1(2)) = false;

    Rect pad1_rect(
            pad_offset1.width,
            pad_offset1.height,
            _bg_area.width - 2 * pad_offset1.width,
            _bg_area.height - 2 * pad_offset1.height
    );

    pad1_rect.width = fmax(pad1_rect.width, 1);
    pad1_rect.height = fmax(pad1_rect.height, 1);
    bg_mask(pad1_rect) = false;

    ////////////////////////////////////////////////////////////////////////
    Size pad_offset2;

    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset2.width = (_bg_area.width - _fg_area.width) / 2;
    pad_offset2.height = (_bg_area.height - _fg_area.height) / 2;


    // difference between bg_area and fg_area has to be even
    if (
            (
                    (pad_offset2.width == round(pad_offset2.width)) &&
                    (pad_offset2.height != round(pad_offset2.height))
            ) ||
            (
                    (pad_offset2.width != round(pad_offset2.width)) &&
                    (pad_offset2.height == round(pad_offset2.height))
            ))
    {
        std::cout<<"difference between bg_area and fg_area has to be even"<<std::endl;
        std::cout<<"pad_offset2.width:"<<pad_offset2.width<<", "
                     <<"pad_offset2.height:"<<pad_offset2.height<<std::endl;

        assert(0);
    }

    pad_offset2.width = fmax(pad_offset2.width, 1);
    pad_offset2.height = fmax(pad_offset2.height, 1);

    if (_DEBUG_OUTPUT) {
        std::cout << "pad_offset2 " << pad_offset2 << std::endl;
    }

    cv::Mat fg_mask(_bg_area, CV_8UC1, cv::Scalar(0)); // init fg_mask

    // xxx: fg_mask(pad_offset2(1)+1:end-pad_offset2(1), pad_offset2(2)+1:end-pad_offset2(2)) = true;

    Rect pad2_rect(
            pad_offset2.width,
            pad_offset2.height,
            _bg_area.width - 2 * pad_offset2.width,
            _bg_area.height - 2 * pad_offset2.height
    );

    fg_mask(pad2_rect) = true;
    ////////////////////////////////////////////////////////////////////////

    cv::Mat fg_mask_new;
    cv::Mat bg_mask_new;

    mexResize(fg_mask, fg_mask_new, _norm_bg_area, "auto");
    mexResize(bg_mask, bg_mask_new, _norm_bg_area, "auto");
//    if (_ORIGINAL_VERSION)
//        depResize(patch, patch, _norm_bg_area);
//    else
//        resize(patch, patch, _norm_bg_area, 0, 0, _RESIZE_TYPE);

    int imgCount = 1;
    int dims = 3;
    const int sizes[] = {_N_BINS, _N_BINS, _N_BINS};
    const int channels[] = {0, 1, 2};
    float bRange[] = {0, 256};
    float gRange[] = {0, 256};
    float rRange[] = {0, 256};
    const float *ranges[] = {bRange, gRange, rRange};

    if (_grayscale_sequence) {
        dims = 1;
    }

    // (TRAIN) BUILD THE MODEL
    if (is_new_model) {
        cv::calcHist(&patch, imgCount, channels, bg_mask_new, _bg_hist, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, _fg_hist, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        (bgtotal == 0) && (bgtotal = 1);
        _bg_hist = _bg_hist / bgtotal;

        int fgtotal = cv::countNonZero(fg_mask_new);
        (fgtotal == 0) && (fgtotal = 1);
        _fg_hist = _fg_hist / fgtotal;
    } else { // update the model
        cv::MatND bg_hist_tmp;
        cv::MatND fg_hist_tmp;

        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        (bgtotal == 0) && (bgtotal = 1);
        bg_hist_tmp = bg_hist_tmp / bgtotal;

        int fgtotal = cv::countNonZero(fg_mask_new);
        (fgtotal == 0) && (fgtotal = 1);
        fg_hist_tmp = fg_hist_tmp / fgtotal;

        // xxx
        _bg_hist = (1 - learning_rate_pwp) * _bg_hist + learning_rate_pwp * bg_hist_tmp;
        _fg_hist = (1 - learning_rate_pwp) * _fg_hist + learning_rate_pwp * fg_hist_tmp;
    }
}