#ifndef RECOGNITION_ENTRY_H
#define RECOGNITION_ENTRY_H

#include <string>
#include <cv.h>

class CRecognitionEntry
{
  public:
    CRecognitionEntry(std::string UniqueName, unsigned LabelId, std::string Comment = std::string());
    ~CRecognitionEntry();

    void GenerateFeatures(
      const cv::Mat& Image,
      const cv::FeatureDetector& FeatureDetector,
      const cv::DescriptorExtractor& DescriptorExtractor);

    bool GenerateFeaturesSurfAdjuster(
      const cv::Mat& Image,
      unsigned AdjusterMin,
      unsigned AdjusterMax,
      unsigned AdjusterIter,
      double AdjusterLearnRate,
      double& HessianThreshold,
      const CvSURFParams* mpSurfParams,
      const cv::DescriptorExtractor& DescriptorExtractor);

    bool GenerateFeaturesGrid(
      const cv::Mat& Image,
      unsigned AdjusterMin,
      unsigned AdjusterMax,
      unsigned AdjusterIter,
      double AdjusterLearnRate,
      double& Threshold,
      const CvSURFParams* mpSurfParams,
      int GridStep,
      const cv::DescriptorExtractor& DescriptorExtractor);

    void GenerateColorHist(const cv::Mat& ImageBGR, unsigned Bins);

    void ShiftKeyPoints(double ShiftX, double ShiftY);

    unsigned GetKeyPointCount() const;
    const std::vector<cv::KeyPoint>& GetKeyPoints() const;
    const cv::Mat& GetDescriptors() const;
    unsigned GetLabelId() const;
    const std::string& GetName() const;
    int GetImageHeight() const;
    int GetImageWidth() const;
    double GetAdjusterThreshold() const;

    const cv::Mat& GetWordHist() const;
    const cv::Mat& GetColorHist() const;

    void IncrementWordHist(unsigned Index);
    void InitWordHist(unsigned Size);
    void NormalizeWordHist(unsigned MaxValue);

    bool LoadFeatures(ifstream& Is);
    bool SaveFeatures(ofstream& Os);

    bool SaveColorHistogram(ofstream& Os);
    bool LoadColorHistogram(ifstream& Is, int ExpectedCols);

  private:
    std::string mName;
    std::string mComment;
    unsigned mLabelId;
    int mImageHeight;
    int mImageWidth;
    double mThreshold; // Hessian threshold (SURF adjuster)

    std::vector<cv::KeyPoint> mKeyPoints;
    cv::Mat mDescriptors;
    cv::Mat mWordHist;
    cv::Mat mColorHist;

};
#endif //end #ifndef RECOGNITION_ENTRY_H