#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace med {
namespace common {

// Very simple Visualizer façade for segmentation/classification demos
class Visualizer {
public:
    // Write a segmentation demo frame (Original | GT | Pred) to a VideoWriter
    //  – targetSize: HxW for each image
    //  – labelHeight: number of pixels above each image reserved for text
    static void writeSegmentationFrame(
        cv::VideoWriter& writer,
        const cv::Mat& original,
        const cv::Mat& groundTruth,
        const cv::Mat& prediction,
        int labelHeight,
        int holdFrames
    );

    // Write a classification demo frame (Original + predicted label + true label)
    static void writeClassificationFrame(
        cv::VideoWriter& writer,
        const cv::Mat& original,
        const std::string& trueLabel,
        const std::string& predLabel,
        int labelHeight,
        int holdFrames
    );
};

} // namespace common
} // namespace med
