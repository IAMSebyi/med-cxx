#include "Benchmark.hpp"

// Assumes that pred and gt have the same size and type CV_8U with values 0 or 255
cv::Mat Benchmark::computeConfusionPixels(const cv::Mat& pred, const cv::Mat& gt) const {
    CV_Assert(pred.size() == gt.size() && pred.type() == CV_8U && gt.type() == CV_8U);
    int TP = 0, FP = 0, FN = 0, TN = 0;
    for (int i = 0; i < pred.rows; ++i) {
        for (int j = 0; j < pred.cols; ++j) {
            int pVal = pred.at<int>(i, j);
            int gVal = gt.at<int>(i, j);
            if (pVal == 255 && gVal == 255) TP++;
            else if (pVal == 255 && gVal == 0) FP++;
            else if (pVal == 0 && gVal == 255) FN++;
            else if (pVal == 0 && gVal == 0) TN++;
        }
    }
    return (cv::Mat_<int>(2, 2) << TP, FP, FN, TN);
}

double Benchmark::computeAccuracyPixels(const cv::Mat& pred, const cv::Mat& gt) const {
    cv::Mat confusion = computeConfusionPixels(pred, gt);
    int TP = confusion.at<int>(0, 0);
    int FP = confusion.at<int>(0, 1);
    int FN = confusion.at<int>(1, 0);
    int TN = confusion.at<int>(1, 1);
    int total = TP + FP + FN + TN;
    return total > 0 ? static_cast<double>(TP + TN) / total : 0.0;
}

double Benchmark::computePrecisionPixels(const cv::Mat& pred, const cv::Mat& gt) const {
    cv::Mat confusion = computeConfusionPixels(pred, gt);
    int TP = confusion.at<int>(0, 0);
    int FP = confusion.at<int>(0, 1);
    return (TP + FP) > 0 ? static_cast<double>(TP) / (TP + FP) : 0.0;
}

double Benchmark::computeRecallPixels(const cv::Mat& pred, const cv::Mat& gt) const {
    cv::Mat confusion = computeConfusionPixels(pred, gt);
    int TP = confusion.at<int>(0, 0);
    int FN = confusion.at<int>(1, 0);
    return (TP + FN) > 0 ? static_cast<double>(TP) / (TP + FN) : 0.0;
}

double Benchmark::computeF1Pixels(const cv::Mat& pred, const cv::Mat& gt) const {
    double precision = computePrecisionPixels(pred, gt);
    double recall = computeRecallPixels(pred, gt);
    return (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

double Benchmark::computeDicePixels(const cv::Mat& pred, const cv::Mat& gt) const {
    cv::Mat confusion = computeConfusionPixels(pred, gt);
    int TP = confusion.at<int>(0, 0);
    int FP = confusion.at<int>(0, 1);
    int FN = confusion.at<int>(1, 0);
    return (2 * TP) > 0 ? static_cast<double>(2 * TP) / (2 * TP + FP + FN) : 0.0;
}

double Benchmark::computeIoUPixels(const cv::Mat& pred, const cv::Mat& gt) const {
    cv::Mat confusion = computeConfusionPixels(pred, gt);
    int TP = confusion.at<int>(0, 0);
    int FP = confusion.at<int>(0, 1);
    int FN = confusion.at<int>(1, 0);
    return (TP + FP + FN) > 0 ? static_cast<double>(TP) / (TP + FP + FN) : 0.0;
}

// Assumes that pred and gt have the same size and type CV_8U or CV_32F
double Benchmark::computeMAE(const cv::Mat& pred, const cv::Mat& gt) const {
    CV_Assert(pred.size() == gt.size() && pred.type() == gt.type() && (pred.type() == CV_8U || pred.type() == CV_32F));
    cv::Mat predF, gtF;
    pred.convertTo(predF, CV_32F);
    gt.convertTo(gtF, CV_32F);
    cv::Mat diff;
    cv::absdiff(predF, gtF, diff);
    return cv::mean(diff)[0];
}

// Assumes that pred and gt have the same size and type CV_8U with values 0 or 255
double Benchmark::computeHausdorff(const cv::Mat& pred, const cv::Mat& gt) const {
    CV_Assert(pred.size() == gt.size() && pred.type() == CV_8U && gt.type() == CV_8U);

    cv::Mat invPred, invGt;
    cv::bitwise_not(pred, invPred);
    cv::bitwise_not(gt, invGt);

    cv::Mat distPred, distGt;
    cv::distanceTransform(invPred, distPred, cv::DIST_L2, 3);
    cv::distanceTransform(invGt, distGt, cv::DIST_L2, 3);

    double maxDistPred = 0.0;
    for (int r = 0; r < pred.rows; ++r) {
        for (int c = 0; c < pred.cols; ++c) {
            if (pred.at<int>(r, c) == 255) {
                maxDistPred = std::max(maxDistPred, static_cast<double>(distGt.at<float>(r, c)));
            }
        }
    }
    double maxDistGt = 0.0;
    for (int r = 0; r < gt.rows; ++r) {
        for (int c = 0; c < gt.cols; ++c) {
            if (gt.at<int>(r, c) == 255) {
                maxDistGt = std::max(maxDistGt, static_cast<double>(distPred.at<float>(r, c)));
            }
        }
    }
    return std::max(maxDistPred, maxDistGt);
}

// Assume that pred and gt are vectors of equal length, containing binary labels (0 or 1).
std::vector<int> Benchmark::computeConfusionLabels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    CV_Assert(pred.size() == gt.size());
    int TP = 0, FP = 0, FN = 0, TN = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        int p = pred[i];
        int g = gt[i];
        if (p == 1 && g == 1) TP++;
        else if (p == 1 && g == 0) FP++;
        else if (p == 0 && g == 1) FN++;
        else if (p == 0 && g == 0) TN++;
    }
    return {TP, FP, FN, TN};
}

double Benchmark::computeAccuracyLabels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    auto confusion = computeConfusionLabels(pred, gt);
    int TP = confusion[0];
    int FP = confusion[1];
    int FN = confusion[2];
    int TN = confusion[3];
    int total = TP + FP + FN + TN;
    return total > 0 ? static_cast<double>(TP + TN) / total : 0.0;
}

double Benchmark::computePrecisionLabels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    auto confusion = computeConfusionLabels(pred, gt);
    int TP = confusion[0];
    int FP = confusion[1];
    return (TP + FP) > 0 ? static_cast<double>(TP) / (TP + FP) : 0.0;
}

double Benchmark::computeRecallLabels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    auto confusion = computeConfusionLabels(pred, gt);
    int TP = confusion[0];
    int FN = confusion[2];
    return (TP + FN) > 0 ? static_cast<double>(TP) / (TP + FN) : 0.0;
}

double Benchmark::computeF1Labels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    double precision = computePrecisionLabels(pred, gt);
    double recall = computeRecallLabels(pred, gt);
    return (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

double Benchmark::computeDiceLabels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    auto confusion = computeConfusionLabels(pred, gt);
    int TP = confusion[0];
    int FP = confusion[1];
    int FN = confusion[2];
    return (2 * TP) > 0 ? static_cast<double>(2 * TP) / (2 * TP + FP + FN) : 0.0;
}

double Benchmark::computeIoULabels(const std::vector<int>& pred, const std::vector<int>& gt) const {
    auto confusion = computeConfusionLabels(pred, gt);
    int TP = confusion[0];
    int FP = confusion[1];
    int FN = confusion[2];
    return (TP + FP + FN) > 0 ? static_cast<double>(TP) / (TP + FP + FN) : 0.0;
}

// Overloaded operator<< for Benchmark (for demonstration purposes, here we print nothing specific)
std::ostream& operator<<(std::ostream& os, const Benchmark& bm) {
    os << "Benchmark metrics (pixel-based and label-based metrics are available)";
    return os;
}
