#include "RecognitionEntry.h"
#include <iostream>
#include <istream>
#include <ostream>
#include <fstream>

#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;

//=================================================================================================
//=================================================================================================
CRecognitionEntry::CRecognitionEntry(string UniqueName, unsigned LabelId, string Comment)
{
  mName = UniqueName;
  mComment = Comment;
  mLabelId = LabelId;
  mImageHeight = 0;
  mImageWidth = 0;
  mThreshold = 0;

  mDescriptors.data = 0;
  mWordHist.data = 0;
  mColorHist.data = 0;
}

//=================================================================================================
//=================================================================================================
CRecognitionEntry::~CRecognitionEntry()
{
}

//=================================================================================================
//=================================================================================================
unsigned CRecognitionEntry::GetKeyPointCount() const
{
  return mKeyPoints.size();
}

//=================================================================================================
//=================================================================================================
bool InRangeInclusive(
  unsigned Min,
  unsigned Max,
  unsigned Val)
{
  if ((Val >= Min) && (Val <= Max))
  {
    return true;
  }
  return false;
}

//=================================================================================================
//=================================================================================================
void CRecognitionEntry::GenerateFeatures(
  const Mat& Image,
  const FeatureDetector& FeatureDetector,
  const DescriptorExtractor& DescriptorExtractor)
{
  mImageHeight = Image.rows;
  mImageWidth = Image.cols;

  FeatureDetector.detect(Image, mKeyPoints);
  DescriptorExtractor.compute(Image, mKeyPoints, mDescriptors);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionEntry::GenerateFeaturesSurfAdjuster(
  const Mat& Image,
  unsigned AdjusterMin,
  unsigned AdjusterMax,
  unsigned AdjusterIter,
  double AdjusterLearnRate,
  double& Threshold,
  const CvSURFParams* mpSurfParams,
  const DescriptorExtractor& DescriptorExtractor)
{
  bool ReturnStatus = false;

  mImageHeight = Image.rows;
  mImageWidth = Image.cols;

  double Alpha = AdjusterLearnRate;
  double PreviousThreshold;
  int    PreviousPoints;
  const int Mid = AdjusterMin + (AdjusterMax-AdjusterMin)/2;

  const double MaxAllowableThreshold = 15000;
  const double MinAllowableThreshold = 1;
  //const double InitialThreshold = Threshold;

  for (int i = 0; i < (int)AdjusterIter; i++)
  {
      SurfFeatureDetector FeatDet =
      SurfFeatureDetector(Threshold, mpSurfParams->nOctaves, mpSurfParams->nOctaveLayers);
    mKeyPoints.clear();
    FeatDet.detect(Image, mKeyPoints);

    //cout << i << " Thresh: " << Threshold << " size: " << mKeyPoints.size() << "\n";

    // Check to see if in range
    if (InRangeInclusive(AdjusterMin, AdjusterMax, mKeyPoints.size()))
    {
      // Break out early if in range
      ReturnStatus = true;
      break;
    }

    if ((i > 0) && (mKeyPoints.size() < AdjusterMin) && (PreviousPoints > (int)AdjusterMax))
    {
      // Detect overshooting condition (too many points -> too few points)
      //cout << "OVERSHOT: TOO MANY POINTS -> TOO FEW POINTS\n";

      const double PreviousThresholdTemp = PreviousThreshold;
      PreviousThreshold = Threshold;

      // If we overshot then calculate the midpoint threshold
      Threshold = Threshold - (Threshold - PreviousThresholdTemp)/2;

      // Back off on the learning rate
      AdjusterLearnRate = AdjusterLearnRate/2.0;
    }
    else if ((i > 0) && (mKeyPoints.size() > AdjusterMax) && (PreviousPoints < (int)AdjusterMin))
    {
      // Detect overshooting condition (Too few points -> too many points)
      //cout << "OVERSHOT: TOO FEW POINTS -> TOO MANY POINTS\n";

      const double PreviousThresholdTemp = PreviousThreshold;
      PreviousThreshold = Threshold;

      // If we overshot then calculate the midpoint threshold
      Threshold = Threshold + (PreviousThresholdTemp - Threshold)/2;

      // Back off on the learning rate
      AdjusterLearnRate = AdjusterLearnRate/2.0;
    }
    else
    {

      if (i > 0) Alpha = abs(AdjusterLearnRate/(Threshold-PreviousThreshold));

      PreviousThreshold = Threshold;

      const double DeltaPoints = (double)mKeyPoints.size()-(double)Mid;

      Threshold = Threshold + Alpha*DeltaPoints;
      if (Threshold < MinAllowableThreshold) Threshold = MinAllowableThreshold;
      if (Threshold > MaxAllowableThreshold) Threshold = MaxAllowableThreshold;
    }

    PreviousPoints = mKeyPoints.size();
  }

  if (ReturnStatus)
  {
    //cout << "HIIIIIIIIIIIIIIIITTTTTT!\n";
  }
  else
  {
    //cout << "MIIIIIIIIIIIIIIISSSSSSS!\n";
  }

  mThreshold = Threshold;

  // Compute the descriptors
  DescriptorExtractor.compute(Image, mKeyPoints, mDescriptors);

  return ReturnStatus;

  //This does not seem to work!
  //FeatureDetector* pDynamicFD;
  //pDynamicFD = new DynamicAdaptedFeatureDetector(new SurfAdjuster(), 400, 500, 50);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionEntry::GenerateFeaturesGrid(
  const Mat& Image,
  unsigned AdjusterMin,
  unsigned AdjusterMax,
  unsigned AdjusterIter,
  double AdjusterLearnRate,
  double& Threshold,
  const CvSURFParams* pSurfParams,
  int GridStep,
  const DescriptorExtractor& DescriptorExtractor)
{
  vector<CRecognitionEntry> Entries;
  mImageHeight = Image.rows;
  mImageWidth = Image.cols;

  // The dimensions of the sliding window
  const int WindowHeight = GridStep;
  const int WindowWidth = GridStep;

  int DesCount = 0;
  int KeyPointCount = 0;

  for (int i = WindowHeight; i <= mImageHeight; i+=WindowHeight)
  {
    for (int j = WindowWidth; j <= mImageWidth; j+=WindowWidth)
    {
      const int MinX = i-WindowHeight;
      const int MaxX = i;
      const int MinY = j-WindowWidth;
      const int MaxY = j;

      CRecognitionEntry Entry = CRecognitionEntry("Entry", 0);

      //cout << "MinX: " << MinX << " MinY: " << MinY << " MaxX " << MaxX << " MaxY " << MaxY << "\n";
      Mat Roi = Mat(Image, Rect(Point(MinX, MinY), Point(MaxX, MaxY)));
      //cout << "ROI rows: " << Roi.rows << " ROI cols: " << Roi.cols << "\n";

      Entry.GenerateFeaturesSurfAdjuster(
        Roi, AdjusterMin, AdjusterMax, AdjusterIter,
        AdjusterLearnRate, Threshold, pSurfParams, DescriptorExtractor);

      Entry.ShiftKeyPoints((double)MinX, (double)MinY);
      Entries.push_back(Entry);
      DesCount += Entry.GetDescriptors().rows;
      KeyPointCount += Entry.GetKeyPointCount();
    }
  }

  int DesIdx = 0;
  int KeyIdx = 0;
  mDescriptors = Mat(DesCount, Entries.at(0).GetDescriptors().cols, CV_32F);
  mKeyPoints.resize(KeyPointCount);

  for (int i = 0; i < (int)Entries.size(); i++)
  {
    // Copy all of the keypoints over to this entry
    const vector<KeyPoint>& KeyPoints = Entries.at(i).GetKeyPoints();
    for (int j = 0; j < (int)KeyPoints.size(); j++)
    {
      mKeyPoints[KeyIdx] = KeyPoints[j];
      KeyIdx++;
    }

    // Copy all of the descriptors over to the this entry
    const Mat& Des = Entries.at(i).GetDescriptors();
    for (int j = 0; j < Des.rows; j++)
    {
      for (int k = 0; k < mDescriptors.cols; k++)
      {
        mDescriptors.at<float>(DesIdx,k) = Des.at<float>(j,k);
      }
      DesIdx++;

      //cout << mDescriptors.row(j) << "\n";
    }
  }

  return true;
}

//=================================================================================================
//=================================================================================================
void CRecognitionEntry::ShiftKeyPoints(double ShiftX, double ShiftY)
{
  for (int i = 0; i < (int)mKeyPoints.size(); i++)
  {
    mKeyPoints[i].pt.x += (float)ShiftX;
    mKeyPoints[i].pt.y += (float)ShiftY;
  }
}

//=================================================================================================
//=================================================================================================
const vector<KeyPoint>& CRecognitionEntry::GetKeyPoints() const
{
  return (const vector<KeyPoint>&)mKeyPoints;
}

//=================================================================================================
//=================================================================================================
const Mat& CRecognitionEntry::GetDescriptors() const
{
  return (const Mat&)mDescriptors;
}
//=================================================================================================
//=================================================================================================
unsigned CRecognitionEntry::GetLabelId() const
{
  return mLabelId;
}

//=================================================================================================
//=================================================================================================
const string& CRecognitionEntry::GetName() const
{
  return (const string&)mName;
}

//=================================================================================================
//=================================================================================================
int CRecognitionEntry::GetImageHeight() const
{
  return mImageHeight;
}

//=================================================================================================
//=================================================================================================
int CRecognitionEntry::GetImageWidth() const
{
  return mImageWidth;
}

//=================================================================================================
//=================================================================================================
double CRecognitionEntry::GetAdjusterThreshold() const
{
  return mThreshold;
}

//=================================================================================================
//=================================================================================================
void CRecognitionEntry::InitWordHist(unsigned Size)
{
  mWordHist = Mat(1, Size, CV_32S, Scalar(0));
}

//=================================================================================================
//=================================================================================================
void CRecognitionEntry::NormalizeWordHist(unsigned MaxValue)
{
  normalize(mWordHist, mWordHist, 0, MaxValue, CV_MINMAX);
}

//=================================================================================================
//=================================================================================================
void CRecognitionEntry::IncrementWordHist(unsigned Index)
{
  if ((int)Index < mWordHist.cols)
  {
    mWordHist.at<int>(0, Index)++;
  }
}

//=================================================================================================
//=================================================================================================
const Mat& CRecognitionEntry::GetWordHist() const
{
  return (const Mat&)mWordHist;
}

//=================================================================================================
//=================================================================================================
const Mat& CRecognitionEntry::GetColorHist() const
{
  return (const Mat&)mColorHist;
}

//=================================================================================================
//=================================================================================================
void CRecognitionEntry::GenerateColorHist(const Mat& ImageBGR, unsigned Bins)
{
  // Initialize histogram settings
  int HistSize[] = {static_cast<int>(Bins)};
  float Range[] = {0, 256}; //{0, 256} = 0 to 255
  const float *Ranges[] = {Range};
  int ChanB[] = {0};
  int ChanG[] = {1};
  int ChanR[] = {2};

  Mat HistB;
  Mat HistG;
  Mat HistR;

  calcHist(&ImageBGR, 1, ChanB, Mat(), // do not use mask
           HistB, 1, HistSize, Ranges,
           true, // the histogram is uniform
           false);

  calcHist(&ImageBGR, 1, ChanG, Mat(), // do not use mask
           HistG, 1, HistSize, Ranges,
           true, // the histogram is uniform
           false);

  calcHist(&ImageBGR, 1, ChanR, Mat(), // do not use mask
           HistR, 1, HistSize, Ranges,
           true, // the histogram is uniform
           false);

  mColorHist = Mat(1, 3*Bins, CV_32F);

  for (unsigned i = 0; i < Bins; i++)
  {
    mColorHist.at<float>(0,i) = HistB.at<float>(i,0);
    mColorHist.at<float>(0,i+Bins) = HistG.at<float>(i,0);
    mColorHist.at<float>(0,i+2*Bins) = HistR.at<float>(i,0);
  }
}

//=================================================================================================
// Description:
//  Reads the entry from a binary file.
//  IMPORTANT: make sure the input file stream is opened with the ios::binary flag
//=================================================================================================
bool CRecognitionEntry::LoadFeatures(ifstream& Is)
{
  int Rows = 0;
  int Cols = 0;

  Is.read(reinterpret_cast<char*>(&Rows), sizeof(Rows));
  Is.read(reinterpret_cast<char*>(&Cols), sizeof(Cols));

  Is.read(reinterpret_cast<char*>(&mImageHeight), sizeof(mImageHeight));
  Is.read(reinterpret_cast<char*>(&mImageWidth), sizeof(mImageWidth));

  mKeyPoints.clear();
  mKeyPoints.resize(Rows);

  mDescriptors = Mat(Rows, Cols, CV_32F);

  for (unsigned i=0; i < mKeyPoints.size(); i++)
  {
    //Assign shorter names for readability
    float& Angle    = mKeyPoints.at(i).angle;
    int& ClassId    = mKeyPoints.at(i).class_id;
    int& Octave     = mKeyPoints.at(i).octave;
    float& X        = mKeyPoints.at(i).pt.x;
    float& Y        = mKeyPoints.at(i).pt.y;
    float& Response = mKeyPoints.at(i).response;
    float& Size     = mKeyPoints.at(i).size;

    //Type& Value
    Is.read(reinterpret_cast<char*>(&Angle),       sizeof(Angle));
    Is.read(reinterpret_cast<char*>(&ClassId),     sizeof(ClassId));
    Is.read(reinterpret_cast<char*>(&Octave),      sizeof(Octave));
    Is.read(reinterpret_cast<char*>(&X),           sizeof(X));
    Is.read(reinterpret_cast<char*>(&Y),           sizeof(Y));
    Is.read(reinterpret_cast<char*>(&Response),    sizeof(Response));
    Is.read(reinterpret_cast<char*>(&Size),        sizeof(Size));

    for (int j = 0; j < mDescriptors.cols; j++)
    {
      float Val;
      Is.read(reinterpret_cast<char*>(&(Val)), sizeof(Val));
      mDescriptors.at<float>(i,j) = Val;
    }
  }
  return true;
}

//=================================================================================================
// Description:
//  Writes the entry to a binary file.
//  IMPORTANT: make sure the output file stream is created with the ios::binary flag
//=================================================================================================
bool CRecognitionEntry::SaveFeatures(ofstream& Os)
{
  if (!mKeyPoints.size()) return false;

  int& Rows = mDescriptors.rows;
  int& Cols = mDescriptors.cols;

  int& ImageHeight = mImageHeight;
  int& ImageWidth = mImageWidth;

  Os.write(reinterpret_cast<char*>(&Rows), sizeof(Rows));
  Os.write(reinterpret_cast<char*>(&Cols), sizeof(Cols));

  Os.write(reinterpret_cast<char*>(&ImageHeight), sizeof(ImageHeight));
  Os.write(reinterpret_cast<char*>(&ImageWidth), sizeof(ImageWidth));

  for (unsigned i = 0; i < mKeyPoints.size(); i++)
  {
    //Assign shorter names for readability
    float& Angle    = mKeyPoints.at(i).angle;
    int& ClassId    = mKeyPoints.at(i).class_id;
    int& Octave     = mKeyPoints.at(i).octave;
    float& X        = mKeyPoints.at(i).pt.x;
    float& Y        = mKeyPoints.at(i).pt.y;
    float& Response = mKeyPoints.at(i).response;
    float& Size     = mKeyPoints.at(i).size;

    //Type& Value
    Os.write(reinterpret_cast<char*>(&Angle),       sizeof(Angle));
    Os.write(reinterpret_cast<char*>(&ClassId),     sizeof(ClassId));
    Os.write(reinterpret_cast<char*>(&Octave),      sizeof(Octave));
    Os.write(reinterpret_cast<char*>(&X),           sizeof(X));
    Os.write(reinterpret_cast<char*>(&Y),           sizeof(Y));
    Os.write(reinterpret_cast<char*>(&Response),    sizeof(Response));
    Os.write(reinterpret_cast<char*>(&Size),        sizeof(Size));

    for (int j = 0; j < mDescriptors.cols; j++)
    {
      float Val = mDescriptors.at<float>(i,j);
      Os.write(reinterpret_cast<char*>(&(Val)), sizeof(Val));
    }
  }
  return true;
}

//=================================================================================================
// Description:
//  Writes the entry to a binary file.
//  IMPORTANT: make sure the output file stream is created with the ios::binary flag
//=================================================================================================
bool CRecognitionEntry::SaveColorHistogram(ofstream& Os)
{
  if (!mColorHist.cols) return false;

  int& Cols = mColorHist.cols;

  Os.write(reinterpret_cast<char*>(&Cols), sizeof(Cols));

  for (int i = 0; i < (int)mColorHist.cols; i++)
  {
    float Val = mColorHist.at<float>(0,i);
    Os.write(reinterpret_cast<char*>(&(Val)), sizeof(Val));
  }
  return true;
}

//=================================================================================================
// Description:
//  Reads the entry from a binary file.
//  IMPORTANT: make sure the input file stream is opened with the ios::binary flag
//=================================================================================================
bool CRecognitionEntry::LoadColorHistogram(ifstream& Is, int ExpectedCols)
{
  int Cols = 0;

  Is.read(reinterpret_cast<char*>(&Cols), sizeof(Cols));

  if (Cols != ExpectedCols) return false;

  mColorHist = Mat(1, Cols, CV_32F);

  for (int i=0; i < mColorHist.cols; i++)
  {
    float Val;
    Is.read(reinterpret_cast<char*>(&(Val)), sizeof(Val));
    mColorHist.at<float>(0,i) = Val;
  }
  return true;
}
