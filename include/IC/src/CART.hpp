#ifndef _CART_h_included
#define _CART_h_included
#include "SF.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

namespace IC {

  class CartEntropy : public SF {
  private:

  public:
    double operator() (const vector<size_t> &B) const {
      //Implemmentation not started
      return 0.0;
    }

    size_t size() const {
      //Implemmentation not started
      return 0;
    }

    CartEntropy() {
      MeanSquareError();
    }

    double MeanSquareError() const {

      //Implementation not finished -20190224


      const char *csv_file_name =  "../data/GDS3893.soft";
      //Read in the data file
      cv::Ptr<cv::ml::TrainData> data_set =
        cv::ml::TrainData::loadFromCSV(csv_file_name, // Input file name
          200, // Header lines (ignore this many)
          120, // Responses are (start) at thie column
          121, // Inputs start at this column
          "ord[0-121]" // All 122 columns are ordered
        );
      // Use defaults for delimeter (',') and missch ('?')
      // Verify that we read in what we think.
      //
      int n_samples = data_set->getNSamples();
      if (n_samples == 0) {
        cerr << "Could not read file: " << csv_file_name << endl;
        exit(-1);
      }
      else {
        cout << "Read " << n_samples << " samples from " << csv_file_name << endl;
      }

      // Split the data, so that 90% is train data
      //
      data_set->setTrainTestSplitRatio(0.90, false);
      int n_train_samples = data_set->getNTrainSamples();
      int n_test_samples = data_set->getNTestSamples();
      cout << "Found " << n_train_samples << " Train Samples, and "
        << n_test_samples << " Test Samples" << endl;

      // Create a DTrees classifier.
      //
      cv::Ptr<cv::ml::RTrees> dtree = cv::ml::RTrees::create();
      
      // set parameters
      float _priors[] = { 1.0, 10.0 };
      cv::Mat priors(1, 2, CV_32F, _priors);
      dtree->setMaxDepth(8);
      dtree->setMinSampleCount(10);
      dtree->setRegressionAccuracy(0.01f);
      dtree->setUseSurrogates(false /* true */);
      dtree->setMaxCategories(15);
      dtree->setCVFolds(0 /*10*/); // nonzero causes core dump
      dtree->setUse1SERule(true);
      dtree->setTruncatePrunedTree(true);
      dtree->setPriors( priors );
      dtree->setPriors(cv::Mat()); // ignore priors for now...
      
      // Now train the model
      // NB: we are only using the "train" part of the data set
      //
      dtree->train(data_set);

      // Having successfully trained the data, we should be able
      // to calculate the error on both the training data, as well
      // as the test data that we held out.
      //
      cv::Mat results;
      float train_performance = dtree->calcError(data_set,
        false, // use train data
        results // cv::noArray()
      );
      return train_performance;
    }
  };
}
#endif 
