//=================================================================================================
// Copyright (c) 2011, Paul Filitchkin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of
//      conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright notice, this list of
//      conditions and the following disclaimer in the documentation and/or other materials
//      provided with the distribution.
//
//    * Neither the name of the organization nor the names of its contributors may be used
//      to endorse or promote products derived from this software without specific prior written
//      permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================
#ifndef RECOGNITION_DB_H
#define RECOGNITION_DB_H

#include <string>
#include <map>
#include <cv.h>

// wxWidgets
#include <wx/filename.h>
#include "RecognitionEntry.h"

class TiXmlNode;
class TiXmlElement;
class CvSVM;
struct CvSVMParams;
struct CvSURFParams;
class wxTimeSpan;

//=================================================================================================
//=================================================================================================
class CRecognitionDb
{
  public:

    struct SDirs
    {
      wxFileName mSetupDir;
      wxFileName mImageDir;
      wxFileName mDatabaseDir;
      wxFileName mLogDir;
    };

    enum EFeatureType
    {
      eSIFT = 0,
      eSURF
    };

    CRecognitionDb();
    ~CRecognitionDb();

    bool OnInit(SDirs Dirs, std::string SetupFileName);

    bool PopulateFeatures();
    bool PopulateFeatures(wxTimeSpan& PopulateTime);

    bool PopulateDictionary();
    bool PopulateDictionary(wxTimeSpan& DictionaryTime, wxTimeSpan& WordTime);

    bool PopulateColorHistograms();
    bool PopulateColorHistograms(wxTimeSpan& Time);

    bool TrainWordClassifier();
    bool TrainWordClassifier(wxTimeSpan& TrainWordTime);

    bool TrainColorClassifier();
    bool TrainColorClassifier(wxTimeSpan& Time);

    bool ClassifyEntry(const CRecognitionEntry& Entry, unsigned* Label);
    bool ClassifyImage(const cv::Mat& Image, double& HessianThreshold , unsigned& Label);

    bool ClassifyDbWords(
      const CRecognitionDb& Db,
      bool GenLog);

    bool ClassifyDbWords(
      const CRecognitionDb& Db,
      std::vector<std::string>& Classify,
      std::vector<std::string>& Truth,
      std::map<std::string, unsigned>& MatchCount,
      std::map<std::string, unsigned>& TruthCount,
      wxTimeSpan& ClassifyTime,
      bool GenLog);

    bool ClassifyDbColor(
      const CRecognitionDb& Db, bool GenLog);

    bool ClassifyDbColor(
      const CRecognitionDb& Db,
      std::vector<std::string>& Classify,
      std::vector<std::string>& Truth,
      std::map<std::string, unsigned>& MatchCount,
      std::map<std::string, unsigned>& TruthCount,
      wxTimeSpan& ClassifyTime,
      bool GenLog);

    // Classify the VisClass entry using color and output the result into Label
    // Returns false if any errors are encountered, true otherwise
    bool ClassifyEntryColor(const CRecognitionEntry& Entry, unsigned* Label);

    bool ClassifyEntrySlidingWindow(
      const CRecognitionEntry& Entry,
      const wxFileName& ImageFileName, int Step);

    static void GenImageSquares(
      const wxFileName& ImageDir, unsigned Dim, double Resize);

    static void GenImageXml(
      const wxFileName& TextListofImages);

    // Generates a brief summary of database classification using comma separated value format
    void GenClassifyDbSummaryCsv(
      std::ofstream& Os,
      std::string DbName,
      std::map<std::string, unsigned>& MatchCount,
      std::map<std::string, unsigned>& TruthCount,
      bool WriteHeader);

    // Generate a brief summary of timing results using comma separated value format
    void GenTimingSummaryCsv(
      std::ofstream& Os,
      std::string DbName,
      wxTimeSpan DictionaryTime,
      wxTimeSpan WordHistTime,
      wxTimeSpan FeaturesTime,
      bool WriteHeader);

    void GenWordLogHtml();

    void GetContours(unsigned i);

    const CRecognitionEntry& GetEntry(unsigned i) const;

    std::string GetName() const;
    std::string GetLogDirName() const;
    std::string GetLabel(unsigned Id) const;
    unsigned GetEntryCount() const;
    wxFileName GetImageFileName(unsigned i) const;

    // Given a binary input stream load a stored dictionary
    bool LoadDictionary(std::ifstream& Is);
    // Write the current dictionary to a binary output stream
    bool SaveDictionary(std::ofstream& Os);

  private:

    //These are the top-level directories used by this database
    SDirs mTopLevelDirs;
    //These are the database specific directories (topLevelDir/databaseName)
    SDirs mDbDirs;

    //The name of the database
    std::string mDbName;
    // String that gets appended to the database directory name
    std::string mAppendToDbDir;
    // String that gets appended to the log directory name
    std::string mAppendToLogDir;

    // Label ID to name mapping
    std::map<std::string, unsigned> mLabelToId;

    // Filename of each entry image (indecies match up with mEntries)
    std::vector<wxFileName> mImageFileNames;

    // Array of entries in the database (one entry per image)
    std::vector<CRecognitionEntry> mEntries;

    // Contains each cluster centroid from kmeans dictionary generation
    cv::Mat* mpDictionary;
    cv::Mat* mpLabels;

    // General Feature settings
    EFeatureType mFeatureType;
    bool mAdjusterOn; // Adjuster only works for SURF currently
    unsigned mAdjusterMin;
    unsigned mAdjusterMax;
    unsigned mAdjusterIter;
    bool mAdjusterMemory;
    double mAdjusterLearnRate;
    bool mGridOn;
    int mGridStep;
    bool mGenFeatureLog;
    bool mAutoLevels;
    bool mCacheFeatures;

    // Feature detector and extractor (SIFT and SURF)
    cv::FeatureDetector* mpFeatureDetector;
    cv::DescriptorExtractor* mpDescriptorExtractor;

    // SIFT feature settings
    cv::SIFT::CommonParams* mpSiftCommonParams;
    cv::SIFT::DetectorParams* mpSiftDetectorParams;
    cv::SIFT::DescriptorParams* mpSiftDescriptorParams;

    // SURF features options
    CvSURFParams* mpSurfParams;
    bool mSurfExtended;

    // Dictionary generation parameters
    unsigned mWordCount; // Number of words in the dictionary
    unsigned mWordKMeansIter; // Number of kmeans clustering iterations
    bool mCacheDictionary;
    bool mGenWordLog;

    // Feature word classifier options
    CvSVMParams* mpWordClassifierParams;
    bool mGenWordClassifierLog;

    // Support vector machine word classifier
    CvSVM* mpWordClassifier;

    // Color classifier options
    bool mGenColorHistogramLog;
    unsigned mColorHistogramBins;
    bool mCacheColorHistogram;
    bool mGenColorClassifierLog;
    CvSVMParams* mpColorClassifierParams;

    // Support vector machine color histogram classifier
    CvSVM* mpColorClassifier;

    std::vector<cv::Scalar> mLabelColors;

    bool CreateWordHist(const CRecognitionEntry& Entry, cv::Mat& WordHist);
    bool CreateWordHist(const cv::Mat& Des, cv::Mat& WordHist);

    bool CreateWordHistMask(
      const CRecognitionEntry& Entry,
      cv::Mat& WordHist,
      const cv::Mat& Mask);

    bool CreateWordHistCircular(
      const cv::vector<cv::KeyPoint>& KeyPoints,
      const cv::vector<int>& WordLabels,
      int MinWords, int MaxWords,
      const cv::Point& Center, double& Radius, cv::Mat& WordHist,
      std::vector<cv::Point>& PointsInCircle);

    void CreateColorHist(
      const cv::Mat& ImageBGR,
      unsigned Bins,
      cv::Mat& ColorHist,
      const cv::Mat& Mask);

    bool FillWordHist(CRecognitionEntry& Entry);

    void WriteHistogramImage(wxFileName& SaveFile, const cv::Mat& Values);

    void CreateMask(cv::Mat& Mask, int MinX, int MaxX, int MinY, int MaxY);

    //Helper function to validate the directory paths
    bool GenerateTopLevelDirs();
    bool GenerateDatabaseDirs();

    // Read XML helper functions (for reading settings)
    void ReadFeaturesElement(TiXmlElement* pFeatures);
    void ReadSiftAttributes(TiXmlElement* pSift);
    void ReadSurfAttributes(TiXmlElement* pSurf);
    void ReadHistogramsElement(TiXmlElement* pElement);
    void ReadDictionaryElement(TiXmlElement* pElement);
    void ReadClassifierElement(TiXmlElement* pElement);
    void ReadEntryElement(TiXmlElement* pElement);
    void ReadDisplayElement(TiXmlElement* pEntry);

    // Generic XML helper functions
    std::string ReadTypeAttribute(TiXmlElement* pElement);
    std::string ReadInputAttribute(TiXmlElement* pElement);
    std::string ReadValueAttribute(TiXmlElement* pElement);

    void ReadDoubleValueAttribute(TiXmlElement* pElement, double *pVal);
    void ReadIntValueAttribute(TiXmlElement* pElement, int *pVal);
    void ReadBoolValueAttribute(TiXmlElement* pElement, bool *pVal);

    // Log generation helper functions
    void GenSetupSummaryLog();
    void GenHtmlSvmParams(CvSVMParams* pParams, std::ofstream& Os);
    void GenEntrySummaryLog(std::string HtmlPath);
    void GenFeatureLogImage(
      const cv::Mat& Image,
      const CRecognitionEntry& Entry);
    void GenFeatureLogHtml(
      const CRecognitionEntry& Entry,
      wxTimeSpan& GenTime);
    void GenVisualWordLogImage(const CRecognitionEntry& Entry);
    void GenColorHistLogHtml();
    void GenClassifyDbLog(const wxFileName& Log,
      const std::map<std::string, unsigned>& MatchCount,
      const std::map<std::string, unsigned>& TruthCount,
      const std::vector<std::string>& Classify,
      const std::vector<std::string>& Truth,
      const std::vector<wxTimeSpan>& Times,
      const CRecognitionDb& Db);

    // Log html generation helpter functions
    void GenHtmlHeader(std::ostream& Os, cv::string Title);
    void GenHtmlFooter(std::ostream& Os);
    void GenHtmlTableHeader(std::ostream& Os, unsigned Padding, unsigned Spacing, unsigned Indent=0);
    void GenHtmlTableFooter(std::ostream& Os, unsigned Indent=0);
    void GenHtmlTableRowBeg(std::ostream& Os, unsigned Indent=0);
    void GenHtmlTableRowEnd(std::ostream& Os, unsigned Indent=0);
    void GenHtmlTableDivBeg(std::ostream& Os, unsigned Indent=0);
    void GenHtmlTableDivEnd(std::ostream& Os, unsigned Indent=0);
    void GenHtmlIndent(std::ostream& Os, unsigned Indent=0);
    void GenHtmlTableLine(std::ostream& Os, const std::vector<std::string>& Values, unsigned Indent=0);
    void GenHtmlTableLine(std::ostream& Os, std::string Name , std::string Value , unsigned Indent=0);
    void GenHtmlTableLine(std::ostream& Os, std::string Name , unsigned Value , unsigned Indent=0);
    void GenHtmlTableLine(std::ostream& Os, std::string Name , double Value , unsigned Indent=0);
    void GenHtmlTableLine(std::ostream& Os, std::string Name , bool Value , unsigned Indent=0);

};
#endif //end #ifndef RECOGNITION_DB_H