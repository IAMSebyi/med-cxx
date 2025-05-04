#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Benchmark class provides methods to compute various evaluation metrics
// for both pixel-level segmentation tasks and label-based classification tasks.
class Benchmark {
public:
    Benchmark() = default;

    // Compute confusion matrix for pixel-based evaluation
    // Returns a 2x2 cv::Mat: [ [TP, FP], [FN, TN] ]
    cv::Mat computeConfusionPixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute pixel-based accuracy: (TP + TN) / total pixels
    double computeAccuracyPixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute pixel-based precision: TP / (TP + FP)
    double computePrecisionPixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute pixel-based recall: TP / (TP + FN)
    double computeRecallPixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute pixel-based F1 score: 2*precision*recall / (precision + recall)
    double computeF1Pixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute pixel-based Dice coefficient: 2*TP / (2*TP + FP + FN)
    double computeDicePixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute pixel-based Intersection over Union (IoU): TP / (TP + FP + FN)
    double computeIoUPixels(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute Mean Absolute Error (MAE) between two images (assumed grayscale)
    double computeMAE(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute Hausdorff distance between two binary images
    double computeHausdorff(const cv::Mat& pred, const cv::Mat& gt) const;

    // Compute confusion matrix for label-based evaluation.
    // Returns a vector<int> of size 4: [TP, FP, FN, TN]
    std::vector<int> computeConfusionLabels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Compute label-based accuracy: (TP + TN) / total samples
    double computeAccuracyLabels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Compute label-based precision: TP / (TP + FP)
    double computePrecisionLabels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Compute label-based recall: TP / (TP + FN)
    double computeRecallLabels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Compute label-based F1 score: 2*precision*recall / (precision + recall)
    double computeF1Labels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Compute label-based Dice coefficient: 2*TP / (2*TP + FP + FN)
    double computeDiceLabels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Compute label-based Intersection over Union (IoU): TP / (TP + FP + FN)
    double computeIoULabels(const std::vector<int>& pred, const std::vector<int>& gt) const;

    // Overloaded << operator to print a brief summary
    friend std::ostream& operator<<(std::ostream& os, const Benchmark& bm);
};
