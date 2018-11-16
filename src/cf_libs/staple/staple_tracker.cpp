//
// Created by super on 18-7-20.
//

/*
 * cv::Size(width, height)
 * cv::Point(y, x)
 * cv::Mat(height, width, channels, ... )
 * cv::Mat save by row after row
 *   2d: address = j * width + i
 *   3d: address = j * width * channels + i * channels + k
 * ------------------------------------------------------------
 * row == heigh == Point.y
 * col == width == Point.x
 * Mat::at(Point(x, y)) == Mat::at(y,x)
 */


#include "staple_tracker.hpp"
#include <iomanip>
#include "psr.hpp"
using namespace std;
using namespace cf_tracking;

StapleTracker::StapleTracker(StapleParameters paras, StapleDebug<T>* debug)
        : _isInitialized(false),
          _PADDING(paras.padding),
          _scaleEstimator(0),
          _OUTPUT_SIGMA_FACTOR(static_cast<T>(paras.outputSigmaFactor)),
          _LAMBDA(static_cast<T>(paras.lambda)),
          _LEARNING_RATE(static_cast<T>(paras.learning_rate_cf)),
          _CELL_SIZE(paras.cellSize),
          _TEMPLATE_SIZE(paras.templateSize),
          _PSR_THRESHOLD(static_cast<T>(paras.psrThreshold)),
          _PSR_PEAK_DEL(paras.psrPeakDel),
          _MERGE_FACTOR(paras.merge_factor),
          _MIN_AREA(10),
          _MAX_AREA_FACTOR(0.8),
          _ID("STAPLEcpp"),
          _ENABLE_TRACKING_LOSS_DETECTION(paras.enableTrackingLossDetection),
          _ORIGINAL_VERSION(paras.originalVersion),
          _RESIZE_TYPE(paras.resizeType),
          _USE_CCS(true),
          _debug(debug)
{
    if (paras.enableScaleEstimator)
    {
        ScaleEstimatorParas<T> sp;

        sp.scaleSigmaFactor = static_cast<T>(paras.scaleSigmaFactor);
        sp.scaleStep = static_cast<T>(paras.scaleStep);
        sp.scaleCellSize = paras.scaleCellSize;
        sp.numberOfScales = paras.numberOfScales;
        sp.scaleModelMaxArea = paras.scaleTemplateArea;
        sp.learningRate = static_cast<T>(paras.learning_rate_scale);

        sp.originalVersion = paras.originalVersion;
        sp.useFhogTranspose = paras.useFhogTranspose;
        sp.resizeType = paras.resizeType;
        sp.lambda = static_cast<T>(paras.lambda);
        _scaleEstimator = new ScaleEstimator<T>(sp);
    }
    if(paras.enableColourTracker)
    {
        ColourParameters<T> cp;
        cp.inner_padding = paras.inner_padding;
        cp.learningRate = paras.learning_rate_pwp;
        cp.n_bins = paras.n_bins;
        cp.fixed_area = paras.templateSize*paras.templateSize;
        cp.cell_size = paras.cellSize;
        cp.padding = paras.padding;
        cp.originalVersion = paras.originalVersion;
        _colourTracker = new Colour_Tracker(cp);
    }

    if (paras.useFhogTranspose)
        cvFhog = &piotr::cvFhogT < T, DFC > ;
    else
        cvFhog = &piotr::cvFhog < T, DFC > ;

    if (_USE_CCS)
        calcDft = &cf_tracking::dftCcs;
    else
        calcDft = &cf_tracking::dftNoCcs;

    // init dft
//    cv::Mat initDft = (cv::Mat_<T>(1, 1) << 1);
//    calcDft(initDft, initDft, 0);

    if (CV_MAJOR_VERSION < 3)
    {
        std::cout << "DsstTracker: Using OpenCV Version: " << CV_MAJOR_VERSION << std::endl;
        std::cout << "For more speed use 3.0 or higher!" << std::endl;
    }

}

/******************The implement of reinit/update/updateAt added by superlee**********************/
bool StapleTracker::reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    return reinit_(image, bb);
}

bool StapleTracker::reinit(const cv::Mat& image, cv::Rect_<float>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    return reinit_(image, bb);
}

bool StapleTracker::reinit(const cv::Mat& image, cv::Rect_<double>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    return reinit_(image, bb);
}

bool StapleTracker::update(const cv::Mat& image, cv::Rect_<int>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    if (update_(image, bb) == false)
        return false;

    boundingBox.x = static_cast<int>(round(bb.x));
    boundingBox.y = static_cast<int>(round(bb.y));
    boundingBox.width = static_cast<int>(round(bb.width));
    boundingBox.height = static_cast<int>(round(bb.height));

    return true;
}

bool StapleTracker::update(const cv::Mat& image, cv::Rect_<float>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    if (update_(image, bb) == false)
        return false;

    boundingBox.x = static_cast<float>(bb.x);
    boundingBox.y = static_cast<float>(bb.y);
    boundingBox.width = static_cast<float>(bb.width);
    boundingBox.height = static_cast<float>(bb.height);
    return true;
}

bool StapleTracker::update(const cv::Mat& image, cv::Rect_<double>& boundingBox)
{
    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    if (update_(image, bb) == false)
        return false;

    boundingBox.x = static_cast<double>(bb.x);
    boundingBox.y = static_cast<double>(bb.y);
    boundingBox.width = static_cast<double>(bb.width);
    boundingBox.height = static_cast<double>(bb.height);

    return true;
}

bool StapleTracker::updateAt(const cv::Mat& image, cv::Rect_<int>& boundingBox)
{
    bool isValid = false;

    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    isValid = updateAt_(image, bb);

    boundingBox.x = static_cast<int>(round(bb.x));
    boundingBox.y = static_cast<int>(round(bb.y));
    boundingBox.width = static_cast<int>(round(bb.width));
    boundingBox.height = static_cast<int>(round(bb.height));

    return isValid;
}

bool StapleTracker::updateAt(const cv::Mat& image, cv::Rect_<float>& boundingBox)
{
    bool isValid = false;

    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    isValid = updateAt_(image, bb);

    boundingBox.x = static_cast<float>(bb.x);
    boundingBox.y = static_cast<float>(bb.y);
    boundingBox.width = static_cast<float>(bb.width);
    boundingBox.height = static_cast<float>(bb.height);

    return isValid;
}

bool StapleTracker::updateAt(const cv::Mat& image, cv::Rect_<double>& boundingBox)
{
    bool isValid = false;

    Rect bb = Rect(
            static_cast<T>(boundingBox.x),
            static_cast<T>(boundingBox.y),
            static_cast<T>(boundingBox.width),
            static_cast<T>(boundingBox.height)
    );

    isValid = updateAt_(image, bb);

    boundingBox.x = static_cast<double>(bb.x);
    boundingBox.y = static_cast<double>(bb.y);
    boundingBox.width = static_cast<double>(bb.width);
    boundingBox.height = static_cast<double>(bb.height);

    return isValid;
}

bool StapleTracker::reinit_(const cv::Mat& image, Rect& boundingBox)
{

    boundingBox &= Rect(0, 0, static_cast<T>(image.cols), static_cast<T>(image.rows));
    if(boundingBox.height <= _CELL_SIZE + 1 || boundingBox.width <= _CELL_SIZE + 1){
        std::cout<<"The target is empty"<<std::endl;
        return false;
    }

    Size targetSize;
    targetSize.width = boundingBox.width;
    targetSize.height = boundingBox.height;
    _scale = 1.0;

    if(_colourTracker)
    {
        _colourTracker->reinit(image, boundingBox);
        _templateSz = _colourTracker->getBgArea();
        if (!_ORIGINAL_VERSION) {
            _scale = _colourTracker->getTmpResizeScale();
        }
    } else{
        //templateSz is the same with normal_bg in the original staple
        _templateSz = Size(floor(targetSize.width * (1 + _PADDING)),
                           floor(targetSize.height * (1 + _PADDING)));
        //     saturate to image size
        Size imsize = image.size();
        _templateSz.width = std::min(_templateSz.width, imsize.width - 1);
        _templateSz.height = std::min(_templateSz.height, imsize.height - 1);

        if (!_ORIGINAL_VERSION)
        {
            // resize to fixed side length _TEMPLATE_SIZE to stabilize FPS
            if (_templateSz.height > _templateSz.width)
                _scale = _templateSz.height / _TEMPLATE_SIZE;
            else
                _scale = _templateSz.width / _TEMPLATE_SIZE;
        }
    }

    _templateSz = Size(round(_templateSz.width / _scale), round(_templateSz.height / _scale));
    assert(_templateSz == _colourTracker->getNormalBgArea());
    _pos.x = floor(boundingBox.x) + floor(boundingBox.width * consts::c0_5);
    _pos.y = floor(boundingBox.y) + floor(boundingBox.height * consts::c0_5);

    _baseTargetSz = Size(targetSize.width / _scale, targetSize.height / _scale);
    _templateScaleFactor = 1 / _scale;

    Size templateSzByCells = Size(floor((_templateSz.width) / _CELL_SIZE),
                                  floor((_templateSz.height) / _CELL_SIZE));

    // translation filter output target
//    T outputSigma = sqrt(_templateSz.area() / ((1 + _PADDING) * (1 + _PADDING)))
//                    * _OUTPUT_SIGMA_FACTOR / _CELL_SIZE;
    Size noral_targetsz;
    noral_targetsz = _colourTracker->getNormalTargetSz();
    T outputSigma
            = sqrt(_baseTargetSz.area() ) * _OUTPUT_SIGMA_FACTOR / _CELL_SIZE;

    _y = gaussianShapedLabels2D<T>(outputSigma, templateSzByCells);
//    gaussianResponse(templateSzByCells, outputSigma, _y);
    calcDft(_y, _yf, 0);

    // translation filter hann window
    cv::Mat cosWindowX;
    cv::Mat cosWindowY;
    cosWindowY = hanningWindow<T>(_yf.rows);
    cosWindowX = hanningWindow<T>(_yf.cols);
    _cosWindow = cosWindowY * cosWindowX.t();

    std::shared_ptr<DFC> hfNum(0);
    cv::Mat hfDen;

    if (getTranslationTrainingData(image, hfNum, hfDen, _pos) == false)
        return false;

    _hfNumerator = hfNum;
    _hfDenominator = hfDen;

    if (_scaleEstimator)
    {
        //_scale * _templateScaleFactor is the scale for the target, not the template
        _scaleEstimator->reinit(image, _pos, targetSize,
                                _scale * _templateScaleFactor);
    }

    _lastBoundingBox = boundingBox;
    _isInitialized = true;
    return true;
}
// GAUSSIANRESPONSE create the (fixed) target response of the correlation filter response


bool StapleTracker::getTranslationTrainingData(const cv::Mat& image, std::shared_ptr<DFC>& hfNum,
                                cv::Mat& hfDen, const Point& pos) const
{
    std::shared_ptr<DFC> xt(0);

    if (getTranslationFeatures(image, xt, pos, _scale) == false)
        return false;

    std::shared_ptr<DFC> xtf;

    if (_USE_CCS)
        xtf = DFC::dftFeatures(xt);
    else
        xtf = DFC::dftFeatures(xt, cv::DFT_COMPLEX_OUTPUT);

    hfNum = DFC::mulSpectrumsFeatures(_yf, xtf, true);
    hfDen = DFC::sumFeatures(DFC::mulSpectrumsFeatures(xtf, xtf, true));

    return true;
}

bool StapleTracker::getTranslationFeatures(const cv::Mat& image, std::shared_ptr<DFC>& features,
                            const Point& pos, T scale) const
{
    cv::Mat patch;
    Size patchSize = _templateSz * scale; // Search Region including Padding(BG size)


    if (getSubWindow(image, patch, patchSize, pos) == false)
        return false;

    if (_ORIGINAL_VERSION)
        depResize(patch, patch, _templateSz);
    else
        resize(patch, patch, _templateSz, 0, 0, _RESIZE_TYPE);

    if (_debug != 0)
        _debug->showPatch(patch);

    cv::Mat floatPatch;
    patch.convertTo(floatPatch, CV_32FC(3));

    features.reset(new DFC());
    cvFhog(floatPatch, features, _CELL_SIZE, DFC::numberOfChannels() - 1);

    // append gray-scale image
    if (patch.channels() == 1)
    {
        if (_CELL_SIZE != 1)
            resize(patch, patch, features->channels[0].size(), 0, 0, _RESIZE_TYPE);

        features->channels[DFC::numberOfChannels() - 1] = patch / 255.0 - 0.5;
    }
    else
    {
        if (_CELL_SIZE != 1)
            resize(patch, patch, features->channels[0].size(), 0, 0, _RESIZE_TYPE);

        cv::Mat grayFrame;
        cvtColor(patch, grayFrame, cv::COLOR_BGR2GRAY);
        grayFrame.convertTo(grayFrame, CV_TYPE);
        grayFrame = grayFrame / 255.0 - 0.5;
        features->channels[DFC::numberOfChannels() - 1] = grayFrame;
    }

    DFC::mulFeatures(features, _cosWindow);
    return true;
}

bool StapleTracker::update_(const cv::Mat& image, Rect& boundingBox)
{
    return updateAtScalePos(image, _pos, _scale, boundingBox);
}

bool StapleTracker::updateAt_(const cv::Mat& image, Rect& boundingBox)
{
    boundingBox &= Rect(0, 0, static_cast<T>(image.cols), static_cast<T>(image.rows));
    if(boundingBox.height <= _CELL_SIZE + 1 || boundingBox.width <= _CELL_SIZE + 1){
        std::cout<<"The Tracking target is too small"<<std::endl;
        return false;
    }
    bool isValid = false;
    T scale = 0;
    Point pos(boundingBox.x + boundingBox.width * consts::c0_5,
              boundingBox.y + boundingBox.height * consts::c0_5);

    // caller's box may have a different aspect ratio
    // compared to the _targetSize; use the larger side
    // to calculate scale
    if (boundingBox.width > boundingBox.height)
        scale = boundingBox.width / _baseTargetSz.width;
    else
        scale = boundingBox.height / _baseTargetSz.height;

    isValid = updateAtScalePos(image, pos, scale, boundingBox, true);
    return isValid;
}

bool StapleTracker::updateAtScalePos(const cv::Mat& image, const Point& oldPos, const T oldScale,
                      Rect& boundingBox, const bool update_at)
{
    ++_frameIdx;

    if (!_isInitialized)
        return false;

    T newScale = oldScale;
    Point newPos = oldPos;
    cv::Point2i maxResponseIdx;
    cv::Mat response;

    /*********************Estimation, update the position and scale************************/
    // in case of error return the last box
    boundingBox = _lastBoundingBox;
    if(!update_at){
        if (detectModel(image, response, maxResponseIdx, newPos, newScale) == false)
            return false;
    }


    // return box
    Rect tempBoundingBox;
    tempBoundingBox.width = _baseTargetSz.width * newScale;
    tempBoundingBox.height = _baseTargetSz.height * newScale;
    tempBoundingBox.x = newPos.x - tempBoundingBox.width / 2;
    tempBoundingBox.y = newPos.y - tempBoundingBox.height / 2;

    if (_ENABLE_TRACKING_LOSS_DETECTION)
    {
        if (evalReponse(image, response, maxResponseIdx,
                        tempBoundingBox) == false)
            return false;
    }

    /**************Train and update the model**************/

    if (updateModel(image, newPos, newScale) == false)
        return false;

    boundingBox &= Rect(0, 0, static_cast<T>(image.cols), static_cast<T>(image.rows));
    boundingBox = tempBoundingBox;
    _lastBoundingBox = tempBoundingBox;
    return true;
}

bool StapleTracker::evalReponse(const cv::Mat &image, const cv::Mat& response,
                 const cv::Point2i& maxResponseIdx,
                 const Rect& tempBoundingBox) const
{
    T peakValue = 0;
    T psrClamped = calcPsr(response, maxResponseIdx, _PSR_PEAK_DEL, peakValue);

    if (_debug != 0)
    {
        _debug->showResponse(response, peakValue);
        _debug->setPsr(psrClamped);
    }

    if (psrClamped < _PSR_THRESHOLD)
        return false;

    // check if we are out of image, too small or too large
    Rect imageRect(Point(0, 0), image.size());
    Rect intersection = imageRect & tempBoundingBox;
    double  bbArea = tempBoundingBox.area();
    double areaThreshold = _MAX_AREA_FACTOR * imageRect.area();
    double intersectDiff = std::abs(bbArea - intersection.area());

    if (intersectDiff > 0.01 || bbArea < _MIN_AREA
        || bbArea > areaThreshold)
        return false;

    return true;
}

bool StapleTracker::detectModel(const cv::Mat& image, cv::Mat& response,
                 cv::Point2i& maxResponseIdx, Point& newPos,
                 T& newScale) const
{
    // find translation
    std::shared_ptr<DFC> xt(0);

    if (getTranslationFeatures(image, xt, newPos, newScale) == false)
        return false;

    std::shared_ptr<DFC> xtf;
    if (_USE_CCS)
        xtf = DFC::dftFeatures(xt);
    else
        xtf = DFC::dftFeatures(xt, cv::DFT_COMPLEX_OUTPUT);

    std::shared_ptr<DFC> sampleSpec = DFC::mulSpectrumsFeatures(_hfNumerator, xtf, false);
    cv::Mat sumXtf = DFC::sumFeatures(sampleSpec);
    cv::Mat hfDenLambda = addRealToSpectrum<T>(_LAMBDA, _hfDenominator);
    cv::Mat responseTf;
    if (_USE_CCS)
        divSpectrums(sumXtf, hfDenLambda, responseTf, 0, false);
    else
        divideSpectrumsNoCcs<T>(sumXtf, hfDenLambda, responseTf);

    cv::Mat translationResponse;
    idft(responseTf, translationResponse, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);


    double maxResponse;
    cv::Mat finalResponse;
    if(_colourTracker)
    {
        Size newsz = _colourTracker->getNormDeltaArea();
        Size normal_delta_area = newsz;
        newsz.width = floor(newsz.width / _CELL_SIZE);
        newsz.height = floor(newsz.height / _CELL_SIZE);
        (int(newsz.height) % 2 == 0) && (newsz.height -= 1);
        (int(newsz.width) % 2 == 0) && (newsz.width -= 1);
        cv::Mat cfResponse;
        // Crop square search region (in feature pixels).
        Point center;
//        center.x = floor(translationResponse.cols* consts::c0_5);
//        center.y = floor(translationResponse.rows* consts::c0_5);
//        if(getSubWindow(translationResponse,cfResponse,newsz,center) == false)
//            return false;
        cropFilterResponse(translationResponse, newsz, cfResponse);

        cv::Point maxLoc;
        if (_CELL_SIZE>1)
            depResize(cfResponse, cfResponse, normal_delta_area);
//        mexResize(cfResponse, cfResponse, normal_delta_area, "auto");
        cv::Mat colourResponse;
        _colourTracker->calColourResponse(image,colourResponse);
        mergeResponses(cfResponse,colourResponse,translationResponse);
//        translationResponse = cfResponse;
        cv::minMaxLoc(translationResponse, 0, &maxResponse, 0, &maxLoc);
        float centerx = (1 + normal_delta_area.width) / 2 - 1;
        float centery = (1 + normal_delta_area.height) / 2 - 1;
//        std::cout<<"newScale:"<<newScale<<std::endl;
//        std::cout<<"_colourTracker->getAreaResizeScale()"<<_colourTracker->getAreaResizeScale()<<std::endl;
        newPos.x += (maxLoc.x - centerx) * _colourTracker->getAreaResizeScale();
        newPos.y += (maxLoc.y - centery) * _colourTracker->getAreaResizeScale();
        cv::Size imsize = image.size();
        if(newPos.x <= 0 || newPos.y <= 0 || newPos.x >= imsize.width || newPos.y >= imsize.height)
        {
            std::cout<<"Tracking Object moves out of the image"<<std::endl;
            return false;
        }
        newPos.x = std::max(int(newPos.x), 0);
        newPos.y = std::max(int(newPos.y), 0);
        newPos.x = std::min(int(newPos.x), imsize.width -1);
        newPos.y = std::min(int(newPos.y), imsize.height -1);

        response = translationResponse;
        maxResponseIdx = maxLoc;



    } else{
        cv::Point maxLoc;
        cv::Point subDelta;
        cv::minMaxLoc(translationResponse, 0, &maxResponse, 0, &maxLoc);
        subDelta = maxLoc;

        if (_CELL_SIZE != 1)
            subDelta = subPixelDelta<T>(translationResponse, maxLoc);

        T posDeltaX = (subDelta.x + 1 - floor(translationResponse.cols / consts::c2_0)) * newScale;
        T posDeltaY = (subDelta.y + 1 - floor(translationResponse.rows / consts::c2_0)) * newScale;
        newPos.x += round(posDeltaX * _CELL_SIZE);
        newPos.y += round(posDeltaY * _CELL_SIZE);


        response = translationResponse;
        maxResponseIdx = maxLoc;
    }

    if (_debug != 0)
        _debug->showResponse(response, maxResponse);

    if (_scaleEstimator)
    {
        //find scale
        T tempScale = newScale * _templateScaleFactor;

        if (_scaleEstimator->detectScale(image, newPos,
                                         tempScale) == false)
            return false;

        newScale = tempScale / _templateScaleFactor;
    }




    return true;
}

void StapleTracker::mergeResponses(const cv::Mat &response_cf,
                                   const cv::Mat &response_pwp,
                                   cv::Mat &response,
                                   const char *merge_method) const
{
    assert(!strcmp(merge_method, "const_factor"));
    double alpha = _MERGE_FACTOR;
    //const char *merge_method = cfg.merge_method;

    // MERGERESPONSES interpolates the two responses with the hyperparameter ALPHA
    response = (1 - alpha) * response_cf + alpha * response_pwp;

    // response = (1 - alpha) * response_cf + alpha * response_pwp;
}

void StapleTracker::cropFilterResponse(const cv::Mat &response_cf, Size response_size, cv::Mat& output) const
{
    cv::Size sz = response_cf.size();
    int w = sz.width;
    int h = sz.height;

    // newh and neww must be odd, as we want an exact center

//    std::cout<<"response_cf:"<<response_size   <<std::endl;
    assert(((int(response_size.width) % 2) == 1) && ((int(response_size.height) % 2) == 1));

    int half_width = floor(response_size.width / 2);
    int half_height = floor(response_size.height / 2);
    int center_x = floor(w / 2);
    int center_y = floor(h/2);

    cv::Range i_range(center_x-half_width, center_x + half_width);
    cv::Range j_range(center_y-half_height, center_y + half_height);

//    cv::Range i_range(-half_width, response_size.width - (1 + half_width));
//    cv::Range j_range(-half_height, response_size.height - (1 + half_height));

    std::vector<int> i_mod_range, j_mod_range;

    for (int k = i_range.start; k <= i_range.end; k++) {
        int val = (k - 1 + w) % w;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++) {
        int val = (k - 1 + h) % h;
        j_mod_range.push_back(val);
    }

    float *OUTPUT = new float[int(response_size.width)*int(response_size.height)];

    for (int i = 0; i < response_size.width; i++)
        for (int j = 0; j < response_size.height; j++) {
            int i_idx = i_mod_range[i];
            int j_idx = j_mod_range[j];

            assert((i_idx < w) && (j_idx < h));

            OUTPUT[j*int(response_size.width)+i] = response_cf.at<float>(j_idx,i_idx);
        }

    output = cv::Mat(response_size.height, response_size.width, CV_32FC1, OUTPUT).clone();
    delete[] OUTPUT;
}


bool StapleTracker::updateModel(const cv::Mat& image, const Point& newPos,
                 T newScale)
{
    _pos = newPos;
    _scale = newScale; //The template scale
    std::shared_ptr<DFC> hfNum(0);
    cv::Mat hfDen;

    if (getTranslationTrainingData(image, hfNum, hfDen, _pos) == false)
        return false;

    _hfDenominator = (1 - _LEARNING_RATE) * _hfDenominator + _LEARNING_RATE * hfDen;
    DFC::mulValueFeatures(_hfNumerator, (1 - _LEARNING_RATE));
    DFC::mulValueFeatures(hfNum, _LEARNING_RATE);
    DFC::addFeatures(_hfNumerator, hfNum);

    if(_colourTracker)
    {
        if(_colourTracker->updateColourModel(image, newPos, newScale * _templateScaleFactor) == false)
            return false;
    }

    // update scale model
    if (_scaleEstimator)
    {
        if (_scaleEstimator->updateScale(image, newPos, newScale * _templateScaleFactor) == false)
            return false;
    }

    return true;
}



