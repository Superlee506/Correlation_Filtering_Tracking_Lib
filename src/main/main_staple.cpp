//
// Created by super on 18-7-16.
//
#include <tclap/CmdLine.h>
#include <iostream>
#include "staple_tracker.hpp"
#include "tracker_run.hpp"

class StapleTrackerRun : public TrackerRun
{
public:
    StapleTrackerRun() : TrackerRun("Staplecpp")
    {}

    virtual ~StapleTrackerRun()
    {}

    virtual cf_tracking::CfTracker* parseTrackerParas(TCLAP::CmdLine& cmd, int argc, const char** argv)
    {
        cf_tracking::StapleParameters paras;
        TCLAP::SwitchArg debugOutput("v", "debug", "Output Debug info!", cmd, false);
        TCLAP::SwitchArg originalVersion("", "original_version", "Use the original parameters found in the DSST paper. Performance is close, "
                                                                 "but differences do still exist!", cmd, false);
        TCLAP::ValueArg<int> templateSize("", "para_template_size", "template size", false,
                                          paras.templateSize, "integer", cmd);
        TCLAP::ValueArg<double> padding("", "para_padding", "padding around the target", false,
                                        paras.padding, "double", cmd);
        TCLAP::ValueArg<double> outputSigmaFactor("", "para_output_sigma_factor", "spatial bandwitdh of the target",
                                                  false, paras.outputSigmaFactor, "double", cmd);
        TCLAP::ValueArg<int> cellSize("", "para_cell_size", "cell size of fhog", false, paras.cellSize, "integer", cmd);
        TCLAP::ValueArg<double> lambda("", "para_lambda", "regularization factor", false, paras.lambda, "double", cmd);
        TCLAP::ValueArg<double> interpFactorCf("", "para_interpFactor_cf", "interpolation factor for learning cf",
                                               false, paras.learning_rate_cf, "double", cmd);
        TCLAP::ValueArg<double> interpFactorColour("", "para_interpFactor_colour", "interpolation factor for learning colour",
                                                   false, paras.learning_rate_pwp, "double", cmd);
        TCLAP::ValueArg<double> interpFactorScale("", "para_interpFactor_scale", "interpolation factor for learning scale",
                                                  false, paras.learning_rate_scale, "double", cmd);
        TCLAP::ValueArg<double> mergeFactor("", "para_merge_factor", "merge factor for Cf and Colour Response",
                                            false, paras.merge_factor, "double", cmd);
        TCLAP::ValueArg<double> scaleSigmaFactor("", "para_scale_sigma_factor", "spatial bandwitdh of the target(scale)",
                                                 false, paras.scaleSigmaFactor, "double", cmd);
        TCLAP::ValueArg<double> scaleStep("", "para_scale_step", "scale_step", false, paras.scaleStep, "double", cmd);
        TCLAP::ValueArg<int> scaleCellSize("", "para_scale_cell_size", "cell size of fhog (scale filter)", false, paras.scaleCellSize, "integer", cmd);
        TCLAP::ValueArg<int> numberOfScales("", "para_scale_number", "number of scale steps", false, paras.numberOfScales, "integer", cmd);
        TCLAP::SwitchArg enableTrackingLossDetection("", "para_enable_tracking_loss", "Enable the tracking loss detection!", cmd, paras.enableTrackingLossDetection);
        TCLAP::ValueArg<double> psrThreshold("", "para_psr_threshold",
                                             "if psr is lower than psr threshold, tracking will stop",
                                             false, paras.psrThreshold, "double", cmd);
        TCLAP::ValueArg<int> psrPeakDel("", "para_psr_peak_del", "amount of pixels that are deleted"
                                                                 "for psr calculation around the peak (1 means that a window of 3 by 3 is deleted; 0 means"
                                                                 "that max response is deleted; 2 * peak_del + 1 pixels are deleted)",
                                        false, paras.psrPeakDel, "integer", cmd);

        cmd.parse(argc, argv);

        paras.padding = padding.getValue();
        paras.outputSigmaFactor = outputSigmaFactor.getValue();
        paras.lambda = lambda.getValue();
         paras.learning_rate_cf = interpFactorCf.getValue();
        paras.learning_rate_pwp = interpFactorColour.getValue();
        paras.learning_rate_scale = interpFactorScale.getValue();
        paras.merge_factor = mergeFactor.getValue();
        paras.cellSize = cellSize.getValue();

        paras.scaleSigmaFactor = scaleSigmaFactor.getValue();
        paras.scaleStep = scaleStep.getValue();
        paras.scaleCellSize = scaleCellSize.getValue();
        paras.numberOfScales = numberOfScales.getValue();
        paras.psrThreshold = psrThreshold.getValue();
        paras.psrPeakDel = psrPeakDel.getValue();
        paras.templateSize = templateSize.getValue();
        paras.enableTrackingLossDetection = enableTrackingLossDetection.getValue();

        // use original paper parameters from
        // Danelljan, Martin, et al., "Accurate scale estimation for robust visual tracking," in Proc. BMVC, 2014
        if (originalVersion.getValue())
        {
            paras.padding = static_cast<double>(1);
            paras.outputSigmaFactor = static_cast<double>(1.0 / 16.0);
            paras.lambda = static_cast<double>(0.01);
            paras.learning_rate_cf = static_cast<double>(0.025);
            paras.templateSize = 100;
            paras.cellSize = 1;

            paras.enableTrackingLossDetection = false;
            paras.psrThreshold = 0;
            paras.psrPeakDel = 1;

            paras.enableScaleEstimator = true;
            paras.scaleSigmaFactor = static_cast<double>(0.25);
            paras.scaleStep = static_cast<double>(1.02);
            paras.scaleCellSize = 4;
            paras.numberOfScales = 33;

            paras.originalVersion = true;
            paras.resizeType = cv::INTER_AREA;
        }

        if (debugOutput.getValue())
        {
            setTrackerDebug(&_debug);
            return new cf_tracking::StapleTracker(paras, &_debug);
        }

        return new cf_tracking::StapleTracker(paras);
    }

private:
    cf_tracking::StapleDebug<cf_tracking::StapleTracker::T> _debug;
};

int main(int argc, const char** argv)
{
    StapleTrackerRun mainObj;

    if (!mainObj.start(argc, argv))
        return -1;

    return 0;
}

