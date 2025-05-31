#include "common/Visualizer.hpp"

namespace med {
namespace common {

void Visualizer::writeSegmentationFrame(
    cv::VideoWriter& writer,
    const cv::Mat& original,
    const cv::Mat& groundTruth,
    const cv::Mat& prediction,
    int labelHeight,
    int holdFrames
) {
    if (!writer.isOpened()) return;

    // Assume all mats are same size HxW, single- or 3-channel.
    int H = original.rows, W = original.cols;
    cv::Mat origLabel (H + labelHeight, W, original.type(), cv::Scalar(0,0,0));
    cv::Mat gtLabel (H + labelHeight, W, groundTruth.type(), cv::Scalar(0,0,0));
    cv::Mat prLabel (H + labelHeight, W, prediction.type(), cv::Scalar(0,0,0));

    // Copy images below label
    original.copyTo(origLabel(cv::Rect(0, labelHeight, W, H)));
    groundTruth.copyTo(gtLabel(cv::Rect(0, labelHeight, W, H)));
    prediction.copyTo(prLabel(cv::Rect(0, labelHeight, W, H)));

    // Put text
    cv::putText(origLabel, "Original", {W / 2 - 40, labelHeight / 2 + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    cv::putText(gtLabel, "Ground Truth", {W / 2 - 60, labelHeight / 2 + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    cv::putText(prLabel, "Prediction", {W / 2 - 50, labelHeight / 2 + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    std::vector<cv::Mat> panels = {origLabel, gtLabel, prLabel};
    cv::Mat frame;
    cv::hconcat(panels, frame);

    for (int i = 0; i < holdFrames; ++i) {
        writer.write(frame);
    }
}

void Visualizer::writeClassificationFrame(
    cv::VideoWriter& writer,
    const cv::Mat& original,
    const std::string& trueLabel,
    const std::string& predLabel,
    int labelHeight,
    int holdFrames
) {
    if (!writer.isOpened()) return;

    int H = original.rows, W = original.cols;
    cv::Mat LabelFrame(H + labelHeight, W, original.type(), cv::Scalar(0,0,0));
    original.copyTo(LabelFrame(cv::Rect(0, labelHeight, W, H)));

    std::string txt = "True: " + trueLabel + " | Pred: " + predLabel;
    cv::putText(LabelFrame, txt, {10, labelHeight/2 + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    for (int i = 0; i < holdFrames; ++i) {
        writer.write(LabelFrame);
    }
}

} // namespace common
} // namespace med
