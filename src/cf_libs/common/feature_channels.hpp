//
// Created by super on 18-7-16.
//

/*
    TODO: create class with nonstatic methods and
    overlad operators; this class should completely
    encapsulate an array of cv::Mat objects
    */

#ifndef FHOG_FEATURE_CHANNELS_H_
#define FHOG_FEATURE_CHANNELS_H_

#include "opencv2/core/core.hpp"
#include "math_helper.hpp"
#include <memory>

namespace cf_tracking
{
    template<int NUMBER_OF_CHANNELS, class T>
    class FeatureChannels_
    {
    public:
        static void mulValueFeatures(std::shared_ptr<FeatureChannels_>& m,
            const T value)
        {
            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                m->channels[i] *= value;
        }

        static void addFeatures(std::shared_ptr<FeatureChannels_>& A,
            const std::shared_ptr<FeatureChannels_>& B)
        {
            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                A->channels[i] += B->channels[i];
        }

        static cv::Mat sumFeatures(const std::shared_ptr<FeatureChannels_>& x)
        {
            cv::Mat res = x->channels[0].clone();

            for (int i = 1; i < NUMBER_OF_CHANNELS; ++i)
                res += x->channels[i];

            return res;
        }

        static cv::Mat sumFeaturesInPlace(const std::shared_ptr<FeatureChannels_>& x)
        {
            for (int i = 1; i < NUMBER_OF_CHANNELS; ++i)
                x->channels[0] += x->channels[i];

            return x->channels[0];
        }

        static void mulFeatures(std::shared_ptr<FeatureChannels_>& features,
            const cv::Mat& m)
        {
            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                features->channels[i] = features->channels[i].mul(m);
        }

        static std::shared_ptr<FeatureChannels_> dftFeatures(
            const std::shared_ptr<FeatureChannels_>& features, int flags = 0)
        {
            std::shared_ptr<FeatureChannels_> res(new FeatureChannels_());

            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                cv::dft(features->channels[i], res->channels[i], flags);

            return res;
        }

        static std::shared_ptr<FeatureChannels_> idftFeatures(
            const std::shared_ptr<FeatureChannels_>& features)
        {
            std::shared_ptr<FeatureChannels_> res(new FeatureChannels_());

            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                idft(features->channels[i], res->channels[i], cv::DFT_REAL_OUTPUT | cv::DFT_SCALE, 0);

            return res;
        }

        static T squaredNormFeaturesCcs(const std::shared_ptr<FeatureChannels_>& Af)
        {
            // TODO: this is still slow and used frequently by gaussian
            // correlation => find an equivalent quicker formulation;
            // Note that reshaping and concatenating the mats first
            // and then multiplying them is slower than
            // this current approach!
            int n = Af->channels[0].rows * Af->channels[0].cols;
            T sum_ = 0;
            cv::Mat elemMul;

            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
            {
                mulSpectrums(Af->channels[i], Af->channels[i], elemMul, 0, true);
                sum_ += static_cast<T>(sumRealOfSpectrum<T>(elemMul));
            }

            return sum_ / n;
        }

        static T squaredNormFeaturesNoCcs(const std::shared_ptr<FeatureChannels_>& Af)
        {
            int n = Af->channels[0].rows * Af->channels[0].cols;
            T sum_ = 0;
            cv::Mat elemMul;

            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
            {
                mulSpectrums(Af->channels[i], Af->channels[i], elemMul, 0, true);
                sum_ += static_cast<T>(cv::sum(elemMul)[0]);
            }

            return sum_ / n;
        }

        static std::shared_ptr<FeatureChannels_> mulSpectrumsFeatures(const std::shared_ptr<FeatureChannels_>& Af,
            const std::shared_ptr<FeatureChannels_>& Bf,
            bool conjBf)
        {
            std::shared_ptr<FeatureChannels_> resf(new FeatureChannels_());

            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                mulSpectrums(Af->channels[i], Bf->channels[i], resf->channels[i], 0, conjBf);

            return resf;
        }

        static std::shared_ptr<FeatureChannels_> mulSpectrumsFeatures(const cv::Mat& Af,
            const std::shared_ptr<FeatureChannels_>& Bf,
            bool conjBf = false)
        {
            std::shared_ptr<FeatureChannels_> resf(new FeatureChannels_());

            for (int i = 0; i < NUMBER_OF_CHANNELS; ++i)
                mulSpectrums(Af, Bf->channels[i], resf->channels[i], 0, conjBf);

            return resf;
        }

        static const int numberOfChannels()
        {
            return NUMBER_OF_CHANNELS;
        }

        cv::Mat channels[NUMBER_OF_CHANNELS];
    };

    template <class T>
    using  FhogFeatureChannels = FeatureChannels_ < 31, T > ;

    template <class T>
    using DsstFeatureChannels = FeatureChannels_ < 28, T > ;
}

#endif
