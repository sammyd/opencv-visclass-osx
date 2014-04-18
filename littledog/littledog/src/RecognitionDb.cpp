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
#include "RecognitionDb.h"

//OpenCV
#include <highgui.h>
#include <ml.h>
#include <opencv2/nonfree/nonfree.hpp>

// Other
#include <histLib.h>
#include <tinyxml.h>

// STL
#include <iostream>
#include <fstream>
#include <sstream>

//wxWidgets
#include <wx/filename.h>
#include <wx/dir.h>

using namespace std;
using namespace cv;

//=================================================================================================
//=================================================================================================
CRecognitionDb::CRecognitionDb()
 : mGenFeatureLog(false),
   mAutoLevels(true),
   mFeatureType(eSURF),
   mCacheFeatures(false),
   mAdjusterOn(false),
   mAdjusterMin(400),
   mAdjusterMax(600),
   mAdjusterIter(10),
   mAdjusterMemory(false),
   mAdjusterLearnRate(0.75),
   mGridOn(false),
   mGridStep(512),
   mpFeatureDetector(0),
   mpDescriptorExtractor(0),
   mpSiftCommonParams(0),
   mpSiftDetectorParams(0),
   mpSiftDescriptorParams(0),
   mpSurfParams(0),
   mSurfExtended(false),
   mpDictionary(0),
   mpLabels(0),
   mWordCount(40),
   mWordKMeansIter(1000),
   mGenWordLog(false),
   mCacheDictionary(false),
   mColorHistogramBins(256),
   mCacheColorHistogram(true),
   mGenColorHistogramLog(false),
   mGenWordClassifierLog(false),
   mpWordClassifier(0),
   mpWordClassifierParams(0),
   mGenColorClassifierLog(false),
   mpColorClassifier(0),
   mpColorClassifierParams(0)
{
}

//=================================================================================================
//=================================================================================================
CRecognitionDb::~CRecognitionDb()
{
  // Noticing some very weird behavior: seems like the destructor gets before the object is
  // destroyed and other functions fail as a result
  if (mpFeatureDetector) delete mpFeatureDetector;
  //if (mpSiftCommonParams) delete mpSiftCommonParams;
  //if (mpSiftDetectorParams) delete mpSiftDetectorParams;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::OnInit(SDirs Dirs, string SetupFileName)
{

  mTopLevelDirs = Dirs;
  // Generate top level directories (log folder, database folder)
  // and check that required directories exist (images folder, setup folder)
  if (!GenerateTopLevelDirs()) return false;

  mDbDirs = Dirs;

  wxFileName SetupFilePath = Dirs.mSetupDir;
  SetupFilePath.SetFullName(SetupFileName);

  wxString FullPath = SetupFilePath.GetFullPath();

  TiXmlDocument Doc(FullPath.c_str());

  if (!SetupFilePath.FileExists())
  {
    cerr << "ERROR: setup file does not exist: " << FullPath << "\n";
    return false;
  }

  if (!Doc.LoadFile())
  {
    cerr << "ERROR: could not load the setup file: " << FullPath << "\n"
      << "Make sure the file is properly formatted XML and that it has an <?xml?>"
      << "tag as the first line\n";
    return false;
  }

  //Read the setup file
  TiXmlNode* pDatabase = 0;
  TiXmlNode* pChild;
  string Value;

  // Find the first node labeled database
  for (pChild = Doc.FirstChild(); pChild != 0; pChild = pChild->NextSibling())
  {
    if (pChild->Type() == TiXmlNode::TINYXML_ELEMENT)
    {
      Value = pChild->Value();
      if (Value == "database")
      {
        pDatabase = pChild;
        break;
      }
    }
  }

  if (pDatabase == 0)
  {
    cerr << "ERROR: Did not find the database node!\n";
    return false;
  }

  // Init variables to a known state
  mEntries.clear();
  mLabelColors.clear();

  delete mpDictionary;
  delete mpLabels;
  delete mpFeatureDetector;
  delete mpSiftCommonParams;
  delete mpSiftDetectorParams;
  delete mpSiftDescriptorParams;
  delete mpSurfParams;
  delete mpWordClassifierParams;
  delete mpWordClassifier;
  delete mpColorClassifierParams;
  delete mpColorClassifier;
  //TODO: change dictionary to vocabulary
  mpDictionary = 0;
  mpLabels = 0;
  mpFeatureDetector = 0;
  mpSiftCommonParams = 0;
  mpSiftDetectorParams = 0;
  mpSiftDescriptorParams = 0;
  mpSurfParams = 0;
  mpWordClassifierParams = 0;
  mpWordClassifier = 0;
  mpColorClassifierParams = 0;
  mpColorClassifier = 0;

  double Version = 0;
  for (
    TiXmlAttribute* pAttrib = pDatabase->ToElement()->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "name")
    {
      mDbName = pAttrib->Value();
    }

    if (Name == "version")
    {
      pAttrib->QueryDoubleValue(&Version);
    }

    if (Name == "appendToLogDir")
    {
      mAppendToLogDir = pAttrib->Value();
    }

    if (Name == "appendToDbDir")
    {
      mAppendToDbDir = pAttrib->Value();
    }
  }

  // TODO: Check something about version

  // Now that we have a database name 
  mDbDirs.mDatabaseDir.AppendDir(mDbName + mAppendToDbDir);
  mDbDirs.mLogDir.AppendDir(mDbName + mAppendToLogDir);
  mDbDirs.mImageDir.AppendDir(mDbName);

  // Generate directories specific to this database
  if (!GenerateDatabaseDirs()) return false;

  for (pChild = pDatabase->FirstChild(); pChild != 0; pChild = pChild->NextSibling())
  {
    if (pChild->Type() == TiXmlNode::TINYXML_ELEMENT)
    {
      Value = pChild->Value();

      TiXmlElement* pElement = pChild->ToElement();

      if (Value == "features")
      {
        ReadFeaturesElement(pElement);
      }
      else if (Value == "histograms")
      {
        ReadHistogramsElement(pElement);
      }
      else if (Value == "dictionary")
      {
        ReadDictionaryElement(pElement);
      }
      else if (Value == "classifier")
      {
        ReadClassifierElement(pElement);
      }
      else if (Value == "entry")
      {
        ReadEntryElement(pElement);
      }
      else if (Value == "display")
      {
        ReadDisplayElement(pElement);
      }
      else
      {
        cerr << "WARNING: Unknown tag: " << Value << "\n";
      }
    }
  }

  GenSetupSummaryLog();

  // Everything was successful
  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::GenerateTopLevelDirs()
{

  if (!mTopLevelDirs.mImageDir.IsDir())
  {
    cerr << "ERROR: The specified image path is not a directory\n";
    return false;
  }

  if (!mTopLevelDirs.mSetupDir.IsDir())
  {
    cerr << "ERROR: The specified setup path is not a directory\n";
    return false;
  }

   // Verify that the top level log directory exsits (create if needed)
  const wxString TopLevelLogDir = mTopLevelDirs.mLogDir.GetFullPath();

  if (!wxDirExists(TopLevelLogDir))
  {
    if (!::wxMkdir(TopLevelLogDir))
    {
      cerr << "ERROR: Could not create: " << TopLevelLogDir << "\n";
      return false;
    }
  }

   // Generate the top level database directory if needed
  const wxString TopLevelDatabaseDir = mTopLevelDirs.mDatabaseDir.GetFullPath();

  if (!wxDirExists(TopLevelDatabaseDir))
  {
    if (!::wxMkdir(TopLevelDatabaseDir))
    {
      cerr << "ERROR: Could not create: " << TopLevelDatabaseDir << "\n";
      return false;
    }
  }

  // Everything was successful
  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::GenerateDatabaseDirs()
{

  // Generate the database specific database directory if needed
  const wxString ThisDatabaseDir = mDbDirs.mDatabaseDir.GetFullPath();

  if (!wxDirExists(ThisDatabaseDir))
  {
    if (!::wxMkdir(ThisDatabaseDir))
    {
      cerr << "ERROR: Could not create: " << ThisDatabaseDir << "\n";
      return false;
    }
  }

  // Generate the database specific log directory if needed
  const wxString ThisLogDir = mDbDirs.mLogDir.GetFullPath();

  if (!wxDirExists(ThisLogDir))
  {
    if (!::wxMkdir(ThisLogDir))
    {
      cerr << "ERROR: Could not create: " << ThisLogDir << "\n";
      return false;
    }
  }

  // Everything was successful
  return true;
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadFeaturesElement(TiXmlElement* pFeatures)
{
  string FeatureType;

  FeatureType = ReadTypeAttribute(pFeatures);

  if (FeatureType == "SIFT")
  {
    mFeatureType = eSIFT;

    ReadSiftAttributes(pFeatures);

    mpFeatureDetector = new SiftFeatureDetector(
      *mpSiftDetectorParams,
      *mpSiftCommonParams);

    mpDescriptorExtractor = new SiftDescriptorExtractor(
      *mpSiftDescriptorParams,
      *mpSiftCommonParams);
  }
  else if (FeatureType == "SURF")
  {
    mFeatureType = eSURF;

    ReadSurfAttributes(pFeatures);

    mpFeatureDetector = new SurfFeatureDetector(
      mpSurfParams->hessianThreshold,
      mpSurfParams->nOctaves,
      mpSurfParams->nOctaveLayers);

    mpDescriptorExtractor = new SurfDescriptorExtractor(
      mpSurfParams->nOctaves,
      mpSurfParams->nOctaveLayers,
      mSurfExtended);
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadSiftAttributes(TiXmlElement* pSift)
{
  // Default SIFT parameters
  int Octaves = 4;
  int OctaveLayers = 3;
  int FirstOctave = -1;
  int AngleMode = 0;
  double Threshold = 0.04/static_cast<double>(OctaveLayers)/2.0;
  double EdgeThreshold = 10.0;
  double Magnification = 4.0;
  bool IsNormal = true;
  bool RecalculateAnlges = false;
  bool AutoLevels = false;
  bool GenerateLog = false;

  for (
    TiXmlElement* pElement = pSift->FirstChildElement();
    pElement != 0;
    pElement = pElement->NextSiblingElement())
  {

    string Param = pElement->Value();

    if (Param == "octaves")
    {
      ReadIntValueAttribute(pElement, &Octaves);
    }
    else if (Param == "octaveLayers")
    {
      ReadIntValueAttribute(pElement, &OctaveLayers);
    }
    else if (Param == "firstOctave")
    {
      ReadIntValueAttribute(pElement, &FirstOctave);
    }
    else if (Param == "angleMode")
    {
      ReadIntValueAttribute(pElement, &AngleMode);
    }
    else if (Param == "threshold")
    {
      ReadDoubleValueAttribute(pElement, &Threshold);
    }
    else if (Param == "edgeThreshold")
    {
      ReadDoubleValueAttribute(pElement, &EdgeThreshold);
    }
    else if (Param == "magnification")
    {
      ReadDoubleValueAttribute(pElement, &EdgeThreshold);
    }
    else if (Param == "isNormal")
    {
      ReadBoolValueAttribute(pElement, &IsNormal);
    }
    else if (Param == "recalculateAnlges")
    {
      ReadBoolValueAttribute(pElement, &RecalculateAnlges);
    }
    else if (Param == "autoLevels")
    {
      ReadBoolValueAttribute(pElement, &AutoLevels);
    }
    else if (Param == "generateLog")
    {
      ReadBoolValueAttribute(pElement, &GenerateLog);
    }
    else if (Param == "cache")
    {
      ReadBoolValueAttribute(pElement, &mCacheFeatures);
    }
  }

  mGenFeatureLog = GenerateLog;
  mAutoLevels = AutoLevels;

  // Fill in the SIFT parameters
  mpSiftCommonParams = new SIFT::CommonParams(Octaves, OctaveLayers, FirstOctave, AngleMode);
  mpSiftDetectorParams = new SIFT::DetectorParams(Threshold, EdgeThreshold);
  mpSiftDescriptorParams = new SIFT::DescriptorParams(Magnification, IsNormal, RecalculateAnlges);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadSurfAttributes(TiXmlElement* pSurf)
{
  // Default SURF parameters
  int Octaves = 4;
  int OctaveLayers = 3;
  double Threshold = 4000.0;

  // The member variables are initialized to the defaults
  int AdjusterMin = mAdjusterMin;
  int AdjusterMax = mAdjusterMax;
  int AdjusterIter = mAdjusterIter;
  bool AdjusterMemory = mAdjusterMemory;
  double AdjusterLearnRate = mAdjusterLearnRate;
  bool GridOn = mGridOn;
  int GridStep = mGridStep;

  for (
    TiXmlElement* pElement = pSurf->FirstChildElement();
    pElement != 0;
    pElement = pElement->NextSiblingElement())
  {
    string Param = pElement->Value();

    if (Param == "octaves")
    {
      ReadIntValueAttribute(pElement, &Octaves);
    }
    else if (Param == "octaveLayers")
    {
      ReadIntValueAttribute(pElement, &OctaveLayers);
    }
    else if (Param == "threshold")
    {
      ReadDoubleValueAttribute(pElement, &Threshold);
    }
    else if (Param == "extended")
    {
      ReadBoolValueAttribute(pElement, &mSurfExtended);
    }
    else if (Param == "adjusterOn")
    {
      ReadBoolValueAttribute(pElement, &mAdjusterOn);
    }
    else if (Param == "adjusterMin")
    {
      ReadIntValueAttribute(pElement, &AdjusterMin);
    }
    else if (Param == "adjusterMax")
    {
      ReadIntValueAttribute(pElement, &AdjusterMax);
    }
    else if (Param == "adjusterIter")
    {
      ReadIntValueAttribute(pElement, &AdjusterIter);
    }
    else if (Param == "adjusterMemory")
    {
      ReadBoolValueAttribute(pElement, &AdjusterMemory);
    }
    else if (Param == "adjusterLearnRate")
    {
      ReadDoubleValueAttribute(pElement, &AdjusterLearnRate);
    }
    else if (Param == "gridOn")
    {
      ReadBoolValueAttribute(pElement, &GridOn);
    }
    else if (Param == "gridStep")
    {
      ReadIntValueAttribute(pElement, &GridStep);
    }
    else if (Param == "autoLevels")
    {
      ReadBoolValueAttribute(pElement, &mAutoLevels);
    }
    else if (Param == "generateLog")
    {
      ReadBoolValueAttribute(pElement, &mGenFeatureLog);
    }
    else if (Param == "cache")
    {
      ReadBoolValueAttribute(pElement, &mCacheFeatures);
    }
  }

  // TODO: perform checks to make sure invalid parameters are not set
  mpSurfParams = new CvSURFParams();
  mpSurfParams->hessianThreshold = Threshold;
  mpSurfParams->nOctaveLayers = OctaveLayers;
  mpSurfParams->nOctaves = Octaves;

  mAdjusterMin  = AdjusterMin;
  mAdjusterMax  = AdjusterMax;
  mAdjusterIter = AdjusterIter;
  mAdjusterMemory = AdjusterMemory;
  mAdjusterLearnRate = AdjusterLearnRate;

  mGridOn = GridOn;
  mGridStep = GridStep;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::PopulateFeatures()
{
  wxTimeSpan PopulateTime(0);
  return PopulateFeatures(PopulateTime);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::PopulateFeatures(wxTimeSpan& PopulateTime)
{
  double HessianThreshold = mpSurfParams->hessianThreshold;
  int AdjusterSuccessCount = 0;
  int AdjusterTotalCount = 0;

  // For every filename in the database
  for (unsigned i = 0; i < mImageFileNames.size(); i++)
  {
    // Construct cached entry name
    wxFileName CachedEntryFileName = mDbDirs.mDatabaseDir;
    CachedEntryFileName.SetName(mImageFileNames.at(i).GetName());
    CachedEntryFileName.SetExt("key");

    // Check to see if there is a cached entry
    if (mCacheFeatures && CachedEntryFileName.IsFileReadable())
    {
      wxDateTime StartTime = wxDateTime::UNow();

      // Read the cached entry as a binary file
      ifstream EntryIs(CachedEntryFileName.GetFullPath().c_str(), ios::in|ios::binary);
      if (EntryIs)
      {
        mEntries.at(i).LoadFeatures(EntryIs);
        EntryIs.close();
        wxTimeSpan Duration = wxDateTime::UNow() - StartTime;

        PopulateTime.Add(Duration);
        //cout << "Loaded    " << CachedEntryFileName.GetName();
        //cout << " with " << mEntries.at(i).GetKeyPointCount();
        //cout << " in " << Duration.Format("%M:%S:%l") << "\n";
      }
    }
    else
    {
      // If we have to generate descriptors then make sure we can
      if ((mpFeatureDetector == 0) || (mpDescriptorExtractor == 0))
      {
        cout << "ERROR: Need to generate descriptors, but no detector or extractor was found!\n";
        return false;
      }

      CRecognitionEntry& Entry = mEntries.at(i);
      Mat Image = cv::imread(mImageFileNames.at(i).GetFullPath().ToStdString());
      Mat ImageNorm;
      Mat& ImageRef = Image;
      int imageHeight = Image.size().height;
      int imageWidth = Image.size().width;

      // Time feature generation
      wxDateTime StartTimer = wxDateTime::UNow();

      if (mAutoLevels)
      {
        // Normalize the image histogram
        NormalizeClipImageBGR(Image, ImageNorm, 1.5);
        // Set the reference as the normalized image
        ImageRef = ImageNorm;
      }

      // Generate the features (keypoints + descriptors)
      if (mAdjusterOn && !mGridOn)
      {
        if (!mAdjusterMemory) HessianThreshold = mpSurfParams->hessianThreshold;

        if (mEntries.at(i).GenerateFeaturesSurfAdjuster(
          ImageRef,
          mAdjusterMin,
          mAdjusterMax,
          mAdjusterIter,
          mAdjusterLearnRate,
          HessianThreshold,
          mpSurfParams,
          *mpDescriptorExtractor))
        {
          AdjusterSuccessCount++;
        }
        AdjusterTotalCount++;
      }
      else if (mAdjusterOn && mGridOn)
      {
        mEntries.at(i).GenerateFeaturesGrid(
          ImageRef,
          mAdjusterMin,
          mAdjusterMax,
          mAdjusterIter,
          mAdjusterLearnRate,
          HessianThreshold,
          mpSurfParams,
          mGridStep,
          *mpDescriptorExtractor);
      }
      else
      {
        mEntries.at(i).GenerateFeatures(
          ImageRef,
          *mpFeatureDetector,
          *mpDescriptorExtractor);
      }

      wxDateTime EndTimer = wxDateTime::UNow();
      wxTimeSpan GenTime = EndTimer - StartTimer;

      PopulateTime.Add(GenTime);

      //cout << "Generated " << CachedEntryFileName.GetName();
      //cout << " with " << Entry.GetKeyPointCount();
      //cout << " in " << GenTime.Format("%M:%S:%l") << "\n";

      // Generate HTML log entry and image for this entry (if enabled)
      if (mGenFeatureLog)
      {
        GenFeatureLogImage(ImageRef, Entry);
        GenFeatureLogHtml(Entry, GenTime);
      }

      // Cache features (if enabled)
      if (mCacheFeatures)
      {
        ofstream EntryOs(CachedEntryFileName.GetFullPath().c_str(), ios::out|ios::binary);
        if (EntryOs) Entry.SaveFeatures(EntryOs);
      }
    }
  }

  //cout << "Populated " << mEntries.size() << " entries in: ";
  //cout << PopulateTime.Format("%M:%S:%l") << "\n";

  //if (mAdjusterOn && (AdjusterTotalCount != 0))
  //{
  //  cout << "Adjuster success rate = ";
  //  cout << 100.0*(double)AdjusterSuccessCount/(double)AdjusterTotalCount << "\n";
  //}

  return true;
}

//=================================================================================================
// Overloaded function that ignores timing the operation
//=================================================================================================
bool CRecognitionDb::PopulateDictionary()
{
  wxTimeSpan DictionaryTime(0);
  wxTimeSpan WordHistTime(0);

  return PopulateDictionary(DictionaryTime, WordHistTime);
}

//=================================================================================================
// Create the dictionary for this database or if one exists (caching enabled) read it from disk
//=================================================================================================
bool CRecognitionDb::PopulateDictionary(wxTimeSpan& DictionaryTime, wxTimeSpan& WordHistTime)
{
  // Time this operation
  wxDateTime StartDictionaryTime = wxDateTime::UNow();

  // Cached dictionary name
  wxFileName CachedDictionaryFileName = mDbDirs.mDatabaseDir;
  CachedDictionaryFileName.SetName(mDbName);
  CachedDictionaryFileName.SetExt("dic");

  // If caching is enabled and the cached file is readable
  if (mCacheDictionary && CachedDictionaryFileName.IsFileReadable())
  {
    // Read/load the cached dictionary
    ifstream DictionaryIs(CachedDictionaryFileName.GetFullPath().c_str(), ios::in|ios::binary);
    if (DictionaryIs && LoadDictionary(DictionaryIs))
    {
      DictionaryIs.close();

      // Make sure the cached entry has the correct number of words
      if (mpDictionary->rows == mWordCount)
      {
        // Cached dictionary loaded successfully so end dictionary timer
        DictionaryTime.Add(wxDateTime::UNow() - StartDictionaryTime);

        // Start word histogram timer
        wxDateTime StartWordHistTime = wxDateTime::UNow();

        // Populate the word histograms
        for (unsigned i = 0; i < mEntries.size(); i++)
        {
          // Store the word histogram in the entry
          if (!FillWordHist(mEntries.at(i)))
          {
            cout << "ERROR: Failed to generate word histogram for entry ";
            cout << mEntries.at(i).GetName() << " (" << i << ")\n";
            return false;
          }
        }

        // End word histogram timer
        WordHistTime.Add(wxDateTime::UNow() - StartWordHistTime);

        // We loaded the dictionary from cache and populated word historgrams successfully
        return true;
      }
    }
  }

  // If caching is turned off, there is no cached dictionary, or the cached
  // dictionary is out of date then generate a new dictionary and fill in
  // the word histograms in this database

  // Check to make sure we have entries
  if (mEntries.size() == 0)
  {
    cout << "ERROR: No entries in database!\n";
    return false;
  }

  const unsigned DescriptorLength = mEntries.at(0).GetDescriptors().cols;

  // Iterate through descriptors and count them
  unsigned KeyPointCount = 0;
  for (unsigned i = 0; i < mEntries.size(); i++)
  {
    KeyPointCount += mEntries.at(i).GetKeyPointCount();
  }

  // Create matrix of descriptors for K-means clustering
  // Each row is a descriptor entry
  Mat AllDescriptors = Mat(KeyPointCount, DescriptorLength, CV_32FC1, Scalar(0));

  unsigned l = 0;

  for (unsigned i = 0; i < mEntries.size(); i++)
  {
    const Mat& Des = mEntries.at(i).GetDescriptors();
    for (unsigned j = 0; j < (unsigned)Des.rows; j++)
    {
      // Copy each row
      for (int k = 0; k < Des.cols; k++)
      {
        AllDescriptors.at<float>(l,k) = Des.at<float>(j,k);
      }
      l++;
    }
  }

  int Attempts = 1;
  mpLabels = new Mat(KeyPointCount, 1, CV_32S);
  mpDictionary = new Mat(DescriptorLength, mWordCount, CV_32F);
  TermCriteria TermCrit = TermCriteria(TermCriteria::MAX_ITER, mWordKMeansIter, 0.0f);

  // Create the dictionary (k-means clustering)
  cv::kmeans(
    AllDescriptors,
    mWordCount,
    *mpLabels,
    TermCrit,
    Attempts,
    KMEANS_PP_CENTERS,
    mpDictionary);

  // End dictionary timer
  DictionaryTime.Add(wxDateTime::UNow() - StartDictionaryTime);

  // Start word histogram timer
  wxDateTime StartWordHistTime = wxDateTime::UNow();

  const unsigned RowDim = mEntries.size();
  const unsigned ColDim = mWordCount;
  unsigned idx = 0;

  for (int i = 0; i < (int)RowDim; i++)
  {
    CRecognitionEntry& Entry = mEntries.at(i);
    Entry.InitWordHist(mWordCount);

    // For each key point descriptor update the word histogram
    for (unsigned j = 0; j < Entry.GetKeyPointCount(); j++)
    {
      unsigned ClusterIndex = mpLabels->at<unsigned>(idx, 0);
      Entry.IncrementWordHist(ClusterIndex);
      idx++;
    }
  }

  // End word histogram timer
  DictionaryTime.Add(wxDateTime::UNow() - StartWordHistTime);

  // Save the dictionary into cache
  if (mCacheDictionary)
  {
    ofstream DictionaryOs(CachedDictionaryFileName.GetFullPath().c_str(), ios::out|ios::binary);
    if (DictionaryOs)
    {
      SaveDictionary(DictionaryOs);
      DictionaryOs.close();
    }
  }

  //Delta = wxDateTime::UNow()-StartTime;
  //cout << "Generated Dictionary in ";
  //cout << Delta.Format("%M:%S:%l") << "\n";

  return true;
}

//=================================================================================================
// Overloaded function that ignores timing the operation
//=================================================================================================
bool CRecognitionDb::PopulateColorHistograms()
{
  wxTimeSpan Time(0);
  return PopulateColorHistograms(Time);
}

//=================================================================================================
// For this database generate color histograms for each entry or if cached color histograms\
// exist (caching enabled) read them from disk
//=================================================================================================
bool CRecognitionDb::PopulateColorHistograms(wxTimeSpan& Time)
{
  // Time this operation
  wxDateTime StartTime = wxDateTime::UNow();

  for (unsigned i = 0; i < mImageFileNames.size(); i++)
  {
    CRecognitionEntry& Entry = mEntries.at(i);

    wxFileName CachedColorHistogramFileName = mDbDirs.mDatabaseDir;
    CachedColorHistogramFileName.SetName(mImageFileNames.at(i).GetName());
    CachedColorHistogramFileName.SetExt("col");

    // If caching is enabled and the cached file is readable
    if (mCacheColorHistogram && CachedColorHistogramFileName.IsFileReadable())
    {
      // Read/load the cached color histogram
      ifstream ColorHistogramIs(
        CachedColorHistogramFileName.GetFullPath().c_str(), ios::in|ios::binary);

      const int ExpectedCols = 3*mColorHistogramBins;
      if (ColorHistogramIs.is_open() && Entry.LoadColorHistogram(ColorHistogramIs, ExpectedCols))
      {
        ColorHistogramIs.close();
      }
    }
    else
    {
      Mat Image = cv::imread(mImageFileNames.at(i).GetFullPath().ToStdString());
      Entry.GenerateColorHist(Image, mColorHistogramBins);

       // Logging
      if (mGenColorHistogramLog)
      {
        wxFileName HistImageFileName = mDbDirs.mLogDir;
        wxString HistImageName = Entry.GetName() + ".Color";
        HistImageFileName.SetName(HistImageName);
        HistImageFileName.SetExt("png");
        const int Bins = mColorHistogramBins;
        const int Height = 200;
        const int Edge = 15;
        // Generate color histogram for the current image
        Mat HistImageBGR = Mat(2*Edge + Height, 2*Edge + 3*Bins, CV_8UC3, Scalar(0));

        // Draw the histogram background and labels
        DrawHistBar(HistImageBGR, Bins, Edge, Height);

        const Mat& HistVals = Entry.GetColorHist();
        double maxBGR = 0;
        minMaxLoc(HistVals, 0, &maxBGR, 0, 0);

        Mat HistB = Mat(1, Bins, CV_32F);
        Mat HistG = Mat(1, Bins, CV_32F);
        Mat HistR = Mat(1, Bins, CV_32F);

        for (int i = 0; i < Bins; i++)
        {
          HistB.at<float>(0,i) = (float)Height*HistVals.at<float>(0,i)/(float)maxBGR;
          HistG.at<float>(0,i) = (float)Height*HistVals.at<float>(0,i+Bins)/(float)maxBGR;
          HistR.at<float>(0,i) = (float)Height*HistVals.at<float>(0,i+2*Bins)/(float)maxBGR;
        }

        DrawHistogram(HistB, HistImageBGR, BLUE_N,  Bins, Edge, Height);
        DrawHistogram(HistG, HistImageBGR, GREEN_N, Bins, Edge, Height);
        DrawHistogram(HistR, HistImageBGR, RED_N,   Bins, Edge, Height);

        cv::imwrite(HistImageFileName.GetFullPath().ToStdString(), HistImageBGR);
      }

      // Generate cached color histogram
      if (mCacheColorHistogram)
      {
        ofstream ColorHistogramOs(
          CachedColorHistogramFileName.GetFullPath().c_str(), ios::out|ios::binary);

        if (ColorHistogramOs.is_open())
        {
          Entry.SaveColorHistogram(ColorHistogramOs);
        }
      }
    }
  }

  Time.Add(wxDateTime::UNow() - StartTime);

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::TrainWordClassifier()
{
  wxTimeSpan TrainWordTime(0);
  return TrainWordClassifier(TrainWordTime);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::TrainWordClassifier(wxTimeSpan& TrainWordTime)
{
  // Time this operation
  wxDateTime StartTime = wxDateTime::UNow();

  if (mpWordClassifierParams == 0) return false;

  const unsigned RowDim = mEntries.size();
  const unsigned ColDim = mWordCount;

  //Contains all of the word histogram vectors for the database
  Mat AllWordHist = Mat(RowDim, ColDim, CV_32F);

  //Contains the corresponding category label
  Mat WordLabel = Mat(RowDim, 1, CV_32F, Scalar(0));

  for (unsigned i = 0; i < RowDim; i++)
  {
    const Mat& WordHist = mEntries.at(i).GetWordHist();

    // Make sure the word histogram is the correct size
    if (WordHist.cols != mWordCount) return false;

    // Fill in a row
    for (unsigned j = 0; j < ColDim; j++)
    {
      AllWordHist.at<float>(i,j) = (float)WordHist.at<int>(0,j);
    }

    WordLabel.at<float>(i,0) = (float)mEntries.at(i).GetLabelId();
  }

  // SVM Classifier
  mpWordClassifier = new CvSVM();
  mpWordClassifier->train(AllWordHist, WordLabel, Mat(), Mat(), *mpWordClassifierParams);

  // Stop timer
  TrainWordTime.Add(wxDateTime::UNow()- StartTime);

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::TrainColorClassifier()
{
  wxTimeSpan Time(0);
  return TrainColorClassifier(Time);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::TrainColorClassifier(wxTimeSpan& Time)
{
  // Time this operation
  wxDateTime StartTime = wxDateTime::UNow();

  if (mpColorClassifierParams == 0) return false;

  const unsigned RowDim = mEntries.size();
  const unsigned ColDim = 3*mColorHistogramBins;

  //Contains all of the word histogram vectors for the database
  Mat AllColorHist = Mat(RowDim, ColDim, CV_32F);

  //Contains the corresponding category label
  Mat Label = Mat(RowDim, 1, CV_32F, Scalar(0));

  for (unsigned i = 0; i < RowDim; i++)
  {
    const Mat& ColorHist = mEntries.at(i).GetColorHist();

    if (ColorHist.cols != ColDim) return false;

    // Fill in a row
    for (unsigned j = 0; j < ColDim; j++)
    {
      AllColorHist.at<float>(i,j) = ColorHist.at<float>(0,j);
    }

    Label.at<float>(i,0) = (float)mEntries.at(i).GetLabelId();
  }

  // SVM Classifier
  mpColorClassifier = new CvSVM();
  mpColorClassifier->train(AllColorHist, Label, Mat(), Mat(), *mpColorClassifierParams);

  Time.Add(wxDateTime::UNow()-StartTime);

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyEntryColor(const CRecognitionEntry& Entry, unsigned* Label)
{

  if (Entry.GetColorHist().data == 0)
  {
    cout << "ERROR: Source entry does not have any color histogram data!\n";
    return false;
  }

  *Label = (unsigned)mpColorClassifier->predict(Entry.GetColorHist());

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyEntry(const CRecognitionEntry& Entry, unsigned* Label)
{

  if (Entry.GetDescriptors().data == 0)
  {
    cout << "ERROR: Source entry does not have any descriptor data!\n";
    return false;
  }

  if (mpDictionary->data == 0)
  {
    cout << "ERROR: Classification database has no dictionary!\n";
    return false;
  }

  // Build the word histogram using the database dictionary
  Mat WordHist = Mat(1, mWordCount, CV_32F, Scalar(0));

  CreateWordHist(Entry, WordHist);

  *Label = (unsigned)mpWordClassifier->predict(WordHist);

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyImage(const Mat& Image, double& HessianThreshold , unsigned& Label)
{

  if (mpDictionary->data == 0)
  {
    cout << "ERROR: Classification database has no dictionary!\n";
    return false;
  }

  CRecognitionEntry Entry("Name", 0);

  Entry.GenerateFeaturesSurfAdjuster(
    Image,
    mAdjusterMin,
    mAdjusterMax,
    mAdjusterIter,
    mAdjusterLearnRate,
    HessianThreshold,
    mpSurfParams,
    *mpDescriptorExtractor);

  if (Entry.GetDescriptors().data == 0)
  {
    cout << "ERROR: Source entry does not have any descriptor data!\n";
    return false;
  }

  // Build the word histogram using the database dictionary
  Mat WordHist = Mat(1, mWordCount, CV_32F, Scalar(0));

  CreateWordHist(Entry, WordHist);

  Label = (unsigned)mpWordClassifier->predict(WordHist);

  return true;
}

//=================================================================================================
//=================================================================================================
void UpdateCount(map<string, unsigned>& Count, const string& Value)
{
  map<string,unsigned>::iterator it = Count.find(Value);

  if (it == Count.end())
  {
    Count[Value] = 1;
  }
  else
  {
    Count[Value]++;
  }
}

//=================================================================================================
//=================================================================================================
unsigned GetCount(const map<string, unsigned>& Count, const string& Value)
{
  map<string,unsigned>::const_iterator it = Count.find(Value);

  if (it == Count.end())
  {
    return 0;
  }
  else
  {
    return it->second;
  }
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyDbWords(const CRecognitionDb& Db, bool GenLog)
{
  vector<string> Classify;
  vector<string> Truth;

  map<string, unsigned> MatchCount;
  map<string, unsigned> TruthCount;

  wxTimeSpan ClassifyTime;

  return ClassifyDbWords(
    Db,
    Classify,
    Truth,
    MatchCount,
    TruthCount,
    ClassifyTime,
    GenLog);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyDbWords(
  const CRecognitionDb& Db,
  vector<string>& Classify,
  vector<string>& Truth,
  map<string, unsigned>& MatchCount,
  map<string, unsigned>& TruthCount,
  wxTimeSpan& ClassifyTime,
  bool GenLog)
{
  // Time this operation
  wxDateTime StartTime = wxDateTime::UNow();

  const int EntryCount = Db.GetEntryCount();
  if (EntryCount == 0)
  {
    cout << "ERROR: Source database does not have any entries!\n";
    return false;
  }

  if (mpWordClassifier == 0)
  {
    cout << "ERROR: Source database does not have a word classifier!\n";
    return false;
  }

  MatchCount.clear();
  TruthCount.clear();

  Classify.resize(EntryCount);
  Truth.resize(EntryCount);
  vector<wxTimeSpan> Times(EntryCount);

  //unsigned MatchCount = 0;
  for (int i = 0; i < EntryCount; i++)
  {
    wxDateTime Start = wxDateTime::UNow();

    // Build the word histogram using the database dictionary
    Mat WordHist = Mat(1, mWordCount, CV_32F, Scalar(0));

    const CRecognitionEntry& Entry = Db.GetEntry(i);

    CreateWordHist(Entry, WordHist);

    unsigned LabelIndex = (unsigned)mpWordClassifier->predict(WordHist);

    Classify[i] = GetLabel(LabelIndex);
    Truth[i] = Db.GetLabel(Entry.GetLabelId());
    Times[i] = wxDateTime::UNow()-Start;

    // Keep track of match statistics
    if (Classify[i] == Truth[i])
    {
      UpdateCount(MatchCount, Truth[i]);
    }
    UpdateCount(TruthCount, Truth[i]);
  }

  // ========================== Logging =============================
  wxFileName LogName = mDbDirs.mLogDir;
  const wxString Name = "~Summary.ClassifyDbWords." + Db.GetName() + "vs" + GetName();
  LogName.SetName(Name);
  LogName.SetExt("html");

  GenClassifyDbLog(LogName, MatchCount, TruthCount, Classify, Truth, Times, Db);

  wxDateTime EndTime = wxDateTime::UNow();
  ClassifyTime = EndTime-StartTime;

  cout << "Classified database using words in ";
  cout << ClassifyTime.Format("%M:%S:%l") << "\n";

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyDbColor(const CRecognitionDb& Db, bool GenLog)
{
  vector<string> Classify;
  vector<string> Truth;

  map<string, unsigned> MatchCount;
  map<string, unsigned> TruthCount;

  wxTimeSpan ClassifyTime;

  return ClassifyDbColor(Db, Classify, Truth, MatchCount, TruthCount, ClassifyTime, GenLog);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyDbColor(
  const CRecognitionDb& Db,
  vector<string>& Classify,
  vector<string>& Truth,
  map<string, unsigned>& MatchCount,
  map<string, unsigned>& TruthCount,
  wxTimeSpan& ClassifyTime,
  bool GenLog)
{

  // Time this operation
  wxDateTime StartTime = wxDateTime::UNow();

  const int EntryCount = Db.GetEntryCount();
  if (EntryCount == 0)
  {
    cout << "ERROR: Source database does not have any entries!\n";
    return false;
  }

  if (mpColorClassifier == 0)
  {
    cout << "ERROR: Source database does not have a color classifier!\n";
    return false;
  }

  Classify.resize(EntryCount);
  Truth.resize(EntryCount);
  vector<wxTimeSpan> Times(EntryCount);

  //map<string, unsigned> MatchCount;
  //map<string, unsigned> TruthCount;

  //unsigned MatchCount = 0;
  for (int i = 0; i < EntryCount; i++)
  {
    wxDateTime Start = wxDateTime::UNow();

    // Build the word histogram using the database dictionary
    Mat WordHist = Mat(1, mWordCount, CV_32F, Scalar(0));

    const CRecognitionEntry& Entry = Db.GetEntry(i);

    unsigned LabelIndex;

    ClassifyEntryColor(Entry, &LabelIndex);

    Classify[i] = GetLabel(LabelIndex);
    Truth[i] = Db.GetLabel(Entry.GetLabelId());
    Times[i] = wxDateTime::UNow()-Start;

    // Keep track of match statistics
    if (Classify[i] == Truth[i])
    {
      UpdateCount(MatchCount, Truth[i]);
    }
    UpdateCount(TruthCount, Truth[i]);
  }

  // ========================== Logging =============================
  wxFileName LogName = mDbDirs.mLogDir;
  const wxString Name = "~Summary.ClassifyDbColor." + Db.GetName() + "vs" + GetName();
  LogName.SetName(Name);
  LogName.SetExt("html");

  GenClassifyDbLog(LogName, MatchCount, TruthCount, Classify, Truth, Times, Db);

  wxDateTime EndTime = wxDateTime::UNow();
  ClassifyTime = EndTime-StartTime;

  cout << "Classified database using color histograms in ";
  cout << ClassifyTime.Format("%M:%S:%l") << "\n";

  return true;
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenClassifyDbLog(
  const wxFileName& LogName,
  const map<string, unsigned>& MatchCount,
  const map<string, unsigned>& TruthCount,
  const vector<string>& Classify,
  const vector<string>& Truth,
  const vector<wxTimeSpan>& Times,
  const CRecognitionDb& Db)
{

  ofstream Os;

  Os.open(LogName.GetFullPath().c_str());
  if (!Os.is_open()) return;

  GenHtmlHeader(Os, LogName.GetName().ToStdString());

  Os << "\t<b>Summary</b>\n";

  int TotalMatchCount = 0;
  int TotalTruthCount = 0;
  GenHtmlTableHeader(Os, 2, 2, 2);

  // Print classification within each category
  for (
    map<string,unsigned>::const_iterator it = TruthCount.begin();
    it != TruthCount.end(); ++it)
  {
    const string& Class = it->first;
    const unsigned ThisClassTotalCount = it->second;
    const unsigned ThisClassMatchCount = GetCount(MatchCount, Class);

    TotalTruthCount += ThisClassTotalCount;
    TotalMatchCount += ThisClassMatchCount;

    string Accuracy = wxString::Format("%0.2f %% (%d/%d)",
      100.0*(double)ThisClassMatchCount/(double)ThisClassTotalCount,
      ThisClassMatchCount, ThisClassTotalCount).ToStdString();

    GenHtmlTableLine(Os, Class, Accuracy, 3);
  }

  const int EntryCount = Db.GetEntryCount();

  // Print the total classification accuracy
  string Accuracy = wxString::Format("<b>%0.2f %% (%d/%d) </b>",
    100.0*(double)TotalMatchCount/(double)EntryCount,
    TotalMatchCount, EntryCount).ToStdString();

  GenHtmlTableLine(Os, "<b>Total</b>", Accuracy, 3);
  GenHtmlTableFooter(Os, 2);

  GenHtmlTableHeader(Os, 2, 2, 2);
  vector<string> ColsToPrint(4,"");
  ColsToPrint[0] = "<b>Image</b>";
  ColsToPrint[1] = "<b>Truth</b>";
  ColsToPrint[2] = "<b>Classification</b>";
  ColsToPrint[3] = "<b>Time</b>";

  GenHtmlTableLine(Os, ColsToPrint, 3);

  for (int i = 0; i < EntryCount; i++)
  {

    wxString ImageSource = "<a href='../" + Db.GetLogDirName() + "/" +
      Db.GetImageFileName(i).GetName() + ".Features.html'><img src='../../" +
      Db.GetImageFileName(i).GetFullPath() + "' height='64px'></a>";

    wxString Color;
    if (Truth[i] == Classify[i]) 
    {
      Color = "#22FF22";
    }
    else
    {
      Color = "#FF2222";
    }

    ColsToPrint[0] = ImageSource;
    ColsToPrint[1] = "<font color='" + Color +"'>" + Truth[i] + "</font>";
    ColsToPrint[2] = "<font color='" + Color +"'>" + Classify[i] + "</font>";
    ColsToPrint[3] = Times[i].Format("%M:%S:%l");

    GenHtmlTableLine(Os, ColsToPrint, 1);
  }
  GenHtmlTableFooter(Os, 2);

  Os.close();
}

//=================================================================================================
// Static Helper function for sliding window classification
//=================================================================================================
void VoteForClass(vector<Mat>& Votes, unsigned Class, const Mat& Mask)
{
  if (Class >= Votes.size()) return;

  for (int i = 0; i < Votes[Class].rows; i++)
  {
    for (int j = 0; j < Votes[Class].cols; j++)
    {
      if (Mask.at<unsigned char>(i,j) != 0)
      {
        Votes[Class].at<unsigned char>(i,j)++;
      }
    }
  }
}

//=================================================================================================
// Static Helper function for visualizing results after sliding window classification
//=================================================================================================
void VisualizeVotes(const vector<Mat>& Votes, unsigned Class, Mat& VoteVis)
{
  if (Class >= Votes.size()) return;

  for (int i = 0; i < Votes[Class].rows; i++)
  {
    for (int j = 0; j < Votes[Class].cols; j++)
    {
      int MostVotes = 0;
      int MostVotesClass = 0;

      for (int k = 0; k < (int)Votes.size(); k++)
      {
        if (Votes[k].at<unsigned char>(i,j) > MostVotes)
        {
          MostVotes = Votes[k].at<unsigned char>(i,j);
          MostVotesClass = k;
        }
      }

      if (MostVotesClass == Class)
      {
        VoteVis.at<unsigned char>(i,j) = 1;
      }
      else
      {
        VoteVis.at<unsigned char>(i,j) = 0;
      }
    }
  }
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::ClassifyEntrySlidingWindow(
  const CRecognitionEntry& Entry,
  const wxFileName& ImageFileName,
  int Step)
{

  if (mpDictionary->data == 0)
  {
    cout << "ERROR: Classification database has no dictionary!\n";
    return false;
  }

  if (!ImageFileName.IsFileReadable())
  {
    cout << "ERROR: Input image is not readable\n";
    return false;
  }

  Mat Image;
    Image = cv::imread(ImageFileName.GetFullPath().ToStdString());

  const int Rows = Image.rows;
  const int Cols = Image.cols;

    vector<Mat> Votes(mLabelToId.size());
  // Do not initialize in the vector class constructor because
  // then each entry in the vector will point to a single Mat!
  for (int i = 0; i < (int)Votes.size(); i++)
  {
    // Each entry gets a unique Mat
    Votes[i] = Mat(Image.rows, Image.cols, CV_8U, Scalar(0));
  }

  const vector<KeyPoint>& KeyPoints = Entry.GetKeyPoints();
  const Mat& Des = Entry.GetDescriptors();
  vector<int> WordLabels(KeyPoints.size());
  const int WordCount = mpDictionary->rows; //This is also equal to mWordCount

  for (int i = 0; i < Des.rows; i++)
  {
    float SmallestNorm = FLT_MAX;
    int SmallestNormIndex = 0;

    for (int j = 0; j < WordCount; j++)
    {
      float Norm = 0;
      // Get the distance between each dictionary word and the current descriptor
      for (int k = 0; k < mpDictionary->cols; k++)
      {
        //Quick and dirty (almost) Euclidean norm is used for comparison
        Norm += pow(mpDictionary->at<float>(j,k) - Des.at<float>(i,k), 2);
      }

      if (Norm < SmallestNorm)
      {
        SmallestNorm = Norm;
        SmallestNormIndex = j;
      }
    }
    WordLabels.at(i) = SmallestNormIndex;
  }

  const int EntryWidth = mEntries.at(0).GetImageWidth();
  const int EntryHeight = mEntries.at(0).GetImageHeight();

  double Radius = (double)EntryWidth/2;

  for (int i = 0; i <= Rows; i+=Step)
  {
    for (int j = 0; j <= Cols; j+=Step)
    {

      // Word classification
      Mat WordHist = Mat(1, mWordCount, CV_32F, Scalar(0));
      vector<Point> PointsInCircle;

      Point Center(i,j);
      CreateWordHistCircular(KeyPoints, WordLabels, mAdjusterMin, mAdjusterMax, Center, Radius, WordHist, PointsInCircle);
      unsigned WordPredictedLabel = (unsigned)mpWordClassifier->predict(WordHist);

      Mat Mask = Mat(Rows, Cols, CV_8U, Scalar(0));
      circle(Mask, Point(i,j), Radius, Scalar(0xFF), -1);
      VoteForClass(Votes, WordPredictedLabel, Mask);

      /*
      vector<Point> Hull;
      vector<int> Branch;
      convexHull(Mat(PointsInCircle), Hull);

      Mat OutImage = Mat(Rows, Cols, CV_8U, Scalar(0));

      for (int i = 0; i < Hull.size(); i++)
      {
        circle(OutImage, Hull.at(i), 4, Scalar(0xFF));
      }

      circle(OutImage, Center, Radius, Scalar(0xFF), 2);

      wxString OutImageName = ImageFileName.GetName() + wxString::Format(".%d.%d.jpg", i, j);

      imwrite(OutImageName.c_str(), OutImage);
      */
      //cout << "i = " << i << "  j = " << j << " " << TotalWords << " " << GetLabel(WordPredictedLabel) << "\n";

      //putText(Image, GetLabel(WordPredictedLabel), Point(i, j), FONT_HERSHEY_COMPLEX, 1, Scalar(0xFF,0xFF,0xFF,0),1);
      //circle(Image, Point(i,j), Radius, Scalar(0,0,0,0));

      //wxString MaskImName = wxString::Format("%d,%d.jpg", i, j);
      //imwrite(MaskImName.c_str(), Mask);
    }
  }

  /*
  const int RowWidth = EntryWidth;
  const int ColWidth = EntryHeight;

  for (int i = RowWidth; i <= Rows; i+=Step)
  {
    for (int j = ColWidth; j <= Cols; j+=Step)
    {
      const int MinX = i-RowWidth;
      const int MaxX = i-1;
      const int MinY = j-ColWidth;
      const int MaxY = j-1;

      //cout << MinX << " " << MaxX << " " << MinY << " " << MaxY << "\n";

      Mat Mask = Mat(Rows, Cols, CV_8U, Scalar(0));
      CreateMask(Mask, MinX, MaxX, MinY, MaxY);
      Mat& SubImage = Mat(Image, Rect(Point(MinY, MinX), Point(MaxY, MaxX)));

      Mat ColorHist;
      CreateColorHist(SubImage, mColorHistogramBins, ColorHist, Mat());

      unsigned ColorPredictedLabel = (unsigned)mpColorClassifier->predict(ColorHist);
      VoteForClass(Votes, ColorPredictedLabel, Mask);
    }
  }
  */

  Mat OutputImage = Mat(Rows, Cols, CV_8UC3);

  for (int i = 0; i < (int)Votes.size(); i++)
  {
    Mat ChannelsMerged = Mat(Rows, Cols, CV_8UC3);
    Mat Vis = Mat(Rows, Cols, CV_8U, Scalar(0));

    VisualizeVotes(Votes, i, Vis);

    vector<Mat> Channels;

    string Name = GetLabel(i) + ".jpg";

    Channels.push_back(((unsigned char)mLabelColors[i][0])*Vis);
    Channels.push_back(((unsigned char)mLabelColors[i][1])*Vis);
    Channels.push_back(((unsigned char)mLabelColors[i][2])*Vis);

    merge(Channels, ChannelsMerged);
    OutputImage = OutputImage + ChannelsMerged;
  }

  string OutputImageName = ImageFileName.GetName().ToStdString() + ".jpg";

  OutputImage = 0.25*OutputImage + 0.75*Image;
  imwrite(OutputImageName.c_str(), OutputImage);

  for (int i = 0; i < (int)Votes.size(); i++)
  {
    Mat ChannelsMerged = Mat(Rows, Cols, CV_8UC3);
    Mat Vis = Mat(Rows, Cols, CV_8U, Scalar(0));
  }

  //imwrite(ImageFileName.GetFullName().c_str(), Image);

  return true;
}


//=================================================================================================
//=================================================================================================
bool CRecognitionDb::CreateWordHistCircular(
  const vector<KeyPoint>& KeyPoints, const vector<int>& WordLabels, int MinWords, int MaxWords,
  const Point& Center, double& Radius, Mat& WordHist, vector<Point>& PointsInCircle)
{
  const float CenterX = Center.x;
  const float CenterY = Center.y;
  double Alpha = 0.25;

  // Aim for the midpoint
  const double Mid = cvRound((double)(MaxWords - MinWords)/2.0) + MinWords;

  bool HitTarget = false;

  const int MaxIterations = 100;
  unsigned WordsWithinRaduis = 0;

  for (int i = 0; i < MaxIterations; i++)
  {
    WordsWithinRaduis = 0;
    for (int j = 0; j < (int)KeyPoints.size(); j++)
    {
      const float PointX = KeyPoints.at(j).pt.x;
      const float PointY = KeyPoints.at(j).pt.y;

      if (sqrt(pow(CenterX - PointX, 2) + pow(CenterY - PointY, 2)) < Radius) WordsWithinRaduis++;
    }

    double Error = Mid - WordsWithinRaduis;

    if ((WordsWithinRaduis < MinWords) || (WordsWithinRaduis > MaxWords))
    {
      Radius = Radius + Alpha*Error;
    }
    else
    {
      HitTarget = true;
      break;
    }
  }

  if (!HitTarget)
  {
    return false;
  }

  PointsInCircle.resize(WordsWithinRaduis);
  int j = 0;
  for (int i = 0; i < KeyPoints.size(); i++)
  {
    const float PointX = KeyPoints.at(i).pt.x;
    const float PointY = KeyPoints.at(i).pt.y;

    if (sqrt(pow(CenterX - PointX, 2) + pow(CenterY - PointY, 2)) < Radius)
    {
      WordHist.at<float>(0, WordLabels.at(i))++;
      PointsInCircle[j] = Point(PointX, PointY);
      j++;
    }
  }

  return true;
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenImageSquares(
  const wxFileName& ImageDir, unsigned Dim, double Resize)
{

  if (!ImageDir.IsDir())
  {
    return;
  }

  wxArrayString AllImages;
  wxDir::GetAllFiles(ImageDir.GetFullPath(), &AllImages,"*.JPG");

  for (unsigned idx = 0; idx < AllImages.size(); idx++)
  {

    wxFileName ImageName(AllImages[idx]);
    Mat ImageOrig = cv::imread(ImageName.GetFullPath().ToStdString());
    Mat Image;

    resize(ImageOrig, Image, Size(),Resize,Resize);

    const int Rows = Image.rows;
    const int Cols = Image.cols;

    // The increment to step by
    const int RowStep = Dim;
    const int ColStep = Dim;

    // The dimensions of the sliding window
    const int RowWidth = Dim;
    const int ColWidth = Dim;

    for (int i = RowWidth; i <= Rows; i+=RowStep)
    {
      for (int j = ColWidth; j <= Cols; j+=ColStep)
      {
        const int MinX = i-RowWidth;
        const int MaxX = i;
        const int MinY = j-ColWidth;
        const int MaxY = j;

        Mat SubImage = Mat(Image, Rect(Point(MinY, MinX), Point(MaxY, MaxX)));

        wxString Name = ImageName.GetName() + wxString::Format(".%d.%d.jpg", i, j);
        cv::imwrite(Name.ToStdString(), SubImage);
      }
    }
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenImageXml(
  const wxFileName& TextListofImages)
{

  if (!TextListofImages.IsFileReadable())
  {
    return;
  }

  ifstream Is;
  Is.open(TextListofImages.GetFullPath().c_str());

  ofstream Os("output.xml");

  string Line;

  while (!Is.eof())
  {
    getline(Is, Line);

    Line.erase(remove(Line.begin(), Line.end(), '\r'), Line.end());
    Line.erase(remove(Line.begin(), Line.end(), '\n'), Line.end());

    cout << Line << "\n";

    Os << "<entry label=\"";

    for (int i = 0; i < Line.size(); i++)
    {
      if (Line[i] == '.')
      {
        Os << Line.substr(0, i) << "\" ";
        break;
      }
    }
    Os << "file=\"" << Line << "\"/>\n";
  }

  Is.close();
  Os.close();
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenClassifyDbSummaryCsv(
  ofstream& Os,
  string DbName,
  map<string, unsigned>& MatchCount,
  map<string, unsigned>& TruthCount,
  bool WriteHeader)
{

  if (WriteHeader)
  {
    Os << "Name, ";

    for (map<string,unsigned>::const_iterator it = TruthCount.begin();
      it != TruthCount.end(); ++it)
    {
      const string& Class = it->first;
      Os << Class << ", ";
    }
    Os << "Total\n";
  }

  Os << DbName << ",";

  unsigned TotalTruthCount = 0;
  unsigned TotalMatchCount = 0;
  //Iterate over all class labels in the database
  for (map<string,unsigned>::const_iterator it = TruthCount.begin();
    it != TruthCount.end(); ++it)
  {
    //The first value is the class label string
    const string& Class = it->first;
    // The second entry is the total count of the current class
    const unsigned ThisClassTotalCount = it->second;
    // Now find how many times this class was matched successfully
    unsigned ThisClassMatchCount = 0;
    map<string,unsigned>::const_iterator cit = MatchCount.find(Class);
    if (cit != MatchCount.end()) ThisClassMatchCount = cit->second;

    // Keep track of the total accuracy
    TotalTruthCount += ThisClassTotalCount;
    TotalMatchCount += ThisClassMatchCount;

    string Accuracy = wxString::Format("%0.2f %%",
      100.0*(double)ThisClassMatchCount/(double)ThisClassTotalCount).ToStdString();

    Os << " " << Accuracy << ",";
  }

  string Accuracy = wxString::Format("%0.2f %%",
      100.0*(double)TotalMatchCount/(double)TotalTruthCount).ToStdString();
  Os << " " << Accuracy << "\n";
}



//=================================================================================================
//=================================================================================================
void CRecognitionDb::WriteHistogramImage(wxFileName& SaveFile, const Mat& Values)
{

  // Check to make sure the input is a row vector, if not then return
  if ((Values.rows <= 1) || (Values.cols > 1)) return;

  const unsigned Bins   = Values.rows;
  const unsigned Edge   = 15;
  const unsigned Height = 100;

  // Generate word histogram for the current image
  Mat Image = Mat(2*Edge + Height, 2*Edge + 3*Bins, CV_8UC3, Scalar(0));

  // Draw background and labels
  DrawHistBar(Image, Bins, Edge, Height);

  Mat ValuesNormalized = Values.clone();

  normalize(Values, ValuesNormalized, 0, Height, CV_MINMAX);

  DrawHistogram(
    ValuesNormalized,
    Image,
    Scalar(0xff, 0xff, 0xff, 0),
    Bins,
    Edge,
    Height);

  cv::imwrite(SaveFile.GetFullPath().ToStdString(), Image);
}

//=================================================================================================
// Creates a square mask defined by corners MinX MaxX  MinY, and MaxY where pixels inside of the
// bounds are set to 255 and pixels outside of the mask are set to 0
//=================================================================================================
void CRecognitionDb::CreateMask(Mat& Mask, int MinX, int MaxX, int MinY, int MaxY)
{
  for (int i = 0; i < Mask.rows; i++)
  {
    for (int j = 0; j < Mask.cols; j++)
    {
      if ((i >= MinX) && (i <= MaxX) && (j >= MinY) && (j <= MaxY))
      {
        Mask.at<unsigned char>(i,j) = 255;
      }
      else
      {
        Mask.at<unsigned char>(i,j) = 0;
      }
    }
  }
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::CreateWordHist(const CRecognitionEntry& Entry, Mat& WordHist)
{

  if ((mpDictionary == 0) || (WordHist.type() != CV_32F)) return false;

 // Each row in this matrix is a word
  const int WordCount = mpDictionary->rows; //This is also equal to mWordCount

  // The word descriptors run along the columns
  const int DesSize = mpDictionary->cols;

  if (WordHist.cols != WordCount) return false;

  // Each row in this matrix is a descriptor
  const Mat& Des = Entry.GetDescriptors();

  return CreateWordHist(Des, WordHist);
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::CreateWordHist(const Mat& Des, Mat& WordHist)
{
  const int WordCount = mpDictionary->rows; //This is also equal to mWordCount

  for (int i = 0; i < Des.rows; i++)
  {
    float SmallestNorm = FLT_MAX;
    int SmallestNormIndex = 0;

    for (int j = 0; j < WordCount; j++)
    {
      float Norm = 0;
      // Get the distance between each dictionary word and the current descriptor
      for (int k = 0; k < mpDictionary->cols; k++)
      {
        //Quick and dirty (almost) Euclidean norm is used for comparison
        Norm += pow(mpDictionary->at<float>(j,k) - Des.at<float>(i,k), 2);
      }

      if (Norm < SmallestNorm)
      {
        SmallestNorm = Norm;
        SmallestNormIndex = j;
      }
    }
    WordHist.at<float>(0,SmallestNormIndex)++;
  }

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::CreateWordHistMask(
  const CRecognitionEntry& Entry, Mat& WordHist, const Mat& Mask)
{

  // Make sure input parameters are valid
  if ((mpDictionary == 0) || (WordHist.type() != CV_32F) || (Mask.type() != CV_8U) ||
    (Entry.GetImageHeight() != Mask.rows) || (Entry.GetImageWidth() != Mask.cols) ) return false;

  // Each row in this matrix is a word
  const int WordCount = mpDictionary->rows; //This is also equal to mWordCount

  // The word descriptors run along the columns
  const int DesSize = mpDictionary->cols;

  if (WordHist.cols != WordCount) return false;

  // Each row in this matrix is a descriptor
  const Mat& Des = Entry.GetDescriptors();

  const vector<KeyPoint>& KeyPoints = Entry.GetKeyPoints();

  for (int i = 0; i < Des.rows; i++)
  {
    int x = (int)KeyPoints.at(i).pt.x;
    int y = (int)KeyPoints.at(i).pt.y;

    // Make sure the keypoint is not masked
    if (Mask.at<unsigned char>(x,y) != 0)
    {
      float SmallestNorm = FLT_MAX;
      int SmallestNormIndex = 0;

      for (int j = 0; j < WordCount; j++)
      {
        float Norm = 0;
        // Get the distance between each dictionary word and the current descriptor
        for (int k = 0; k < mpDictionary->cols; k++)
        {
          //Quick and dirty (almost) Euclidean norm is used for comparison
          Norm += pow(mpDictionary->at<float>(j,k) - Des.at<float>(i,k), 2);
        }

        if (Norm < SmallestNorm)
        {
          SmallestNorm = Norm;
          SmallestNormIndex = j;
        }
      }
      WordHist.at<float>(0,SmallestNormIndex)++;
    }
  }

  return true;
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::CreateColorHist(
  const Mat& ImageBGR,
  unsigned Bins,
  Mat& ColorHist,
  const Mat& Mask)
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

  calcHist(&ImageBGR, 1, ChanB, Mask,
           HistB, 1, HistSize, Ranges,
           true, // the histogram is uniform
           false);

  calcHist(&ImageBGR, 1, ChanG, Mask,
           HistG, 1, HistSize, Ranges,
           true, // the histogram is uniform
           false);

  calcHist(&ImageBGR, 1, ChanR, Mask,
           HistR, 1, HistSize, Ranges,
           true, // the histogram is uniform
           false);

  ColorHist = Mat(1, 3*Bins, CV_32F);

  for (unsigned i = 0; i < Bins; i++)
  {
    ColorHist.at<float>(0,i) = HistB.at<float>(i,0);
    ColorHist.at<float>(0,i+Bins) = HistG.at<float>(i,0);
    ColorHist.at<float>(0,i+2*Bins) = HistR.at<float>(i,0);
  }
}

//=================================================================================================
// Given an input entry fill in its BOVW histogram using the dictionary of this database
// The entry must include descriptors
//=================================================================================================
bool CRecognitionDb::FillWordHist(CRecognitionEntry& Entry)
{

  if (mpDictionary == 0) return false;

 // Each row in this matrix is a word
  const int WordCount = mpDictionary->rows; //This is also equal to mWordCount

  // The word descriptors run along the columns
  const int DesSize = mpDictionary->cols;

  Entry.InitWordHist(WordCount);

  // Each row in this matrix is a descriptor
  const Mat& Des = Entry.GetDescriptors();

  if (Des.rows == 0) return false;

  for (int i = 0; i < Des.rows; i++)
  {
    float SmallestNorm = FLT_MAX;
    int SmallestNormIndex = 0;

    for (int j = 0; j < WordCount; j++)
    {
      float Norm = 0;
      // Get the "Euclidean" norm between each dictionary word and the current descriptor
      for (int k = 0; k < mpDictionary->cols; k++)
      {
        //Quick and dirty (almost) Euclidean norm is used for comparison
        Norm += pow(mpDictionary->at<float>(j,k) - Des.at<float>(i,k), 2);
      }

      if (Norm < SmallestNorm)
      {
        SmallestNorm = Norm;
        SmallestNormIndex = j;
      }
    }
    Entry.IncrementWordHist(SmallestNormIndex);
  }

  return true;
}

//=================================================================================================
// This function does not play a fundamental role in classification.
// It was used soley for experimentation
//=================================================================================================
void CRecognitionDb::GetContours(unsigned ImageIndex)
{
  Mat Image = cv::imread(mImageFileNames.at(ImageIndex).GetFullPath().ToStdString());
  Mat Edges;

  cvtColor(Image, Edges, CV_BGR2GRAY);
  GaussianBlur(Edges, Edges, Size(33,33), 4, 4);
  Canny(Edges, Edges, 10, 15, 3);

  //wxFileName OutputFileName = mDbDirs.mLogDir;
  wxString EdgesName = mImageFileNames.at(ImageIndex).GetName() + ".contour.jpg";
  cv::imwrite(EdgesName.ToStdString(), Edges);

 // The dimensions of the sliding window
  const int WindowHeight = 64;
  const int WindowWidth = 64;
  const int ImageHeight = Image.rows;
  const int ImageWidth = Image.cols;

  vector<float> Sums;
  float SumAvg = 0;
  for (int i = WindowHeight; i <= ImageHeight; i+=WindowHeight)
  {
    for (int j = WindowWidth; j <= ImageWidth; j+=WindowWidth)
    {
      const int MinX = i-WindowHeight;
      const int MaxX = i;
      const int MinY = j-WindowWidth;
      const int MaxY = j;

      const Mat& EdgesRoi = Mat(Edges, Rect(Point(MinY, MinX), Point(MaxY, MaxX)));
      const float EdgesRoiAvg = sum(EdgesRoi)[0]/(float)(WindowHeight*WindowWidth);
      Sums.push_back(EdgesRoiAvg);
      SumAvg += EdgesRoiAvg;
    }
  }
  SumAvg = SumAvg/(float)Sums.size();

  Mat Output = Mat(ImageHeight, ImageWidth, CV_8U);

  int k = 0;
  for (int i = WindowHeight; i <= ImageHeight; i+=WindowHeight)
  {
    for (int j = WindowWidth; j <= ImageWidth; j+=WindowWidth)
    {
      const int MinX = i-WindowHeight;
      const int MaxX = i;
      const int MinY = j-WindowWidth;
      const int MaxY = j;

      Mat OutputRoi = Mat(Output, Rect(Point(MinY, MinX), Point(MaxY, MaxX)));

      if (Sums[k] > SumAvg)
      {
        OutputRoi.setTo(Scalar(255));
      }
      else
      {
        OutputRoi.setTo(Scalar(0));
      }
      k++;
    }
  }

  wxString OutputName = mImageFileNames.at(ImageIndex).GetName() + ".avg.jpg";
  cv::imwrite(OutputName.ToStdString(), Output);
}

//=================================================================================================
//=================================================================================================
const CRecognitionEntry& CRecognitionDb::GetEntry(unsigned i) const
{
  return (const CRecognitionEntry&)mEntries.at(i);
}

//=================================================================================================
//=================================================================================================
unsigned CRecognitionDb::GetEntryCount() const
{
  return mEntries.size();
}

//=================================================================================================
//=================================================================================================
std::string CRecognitionDb::GetName() const
{
  return mDbName;
}

//=================================================================================================
//=================================================================================================
std::string CRecognitionDb::GetLogDirName() const
{
  return mDbName + mAppendToLogDir;
}
//=================================================================================================
//=================================================================================================
std::string CRecognitionDb::GetLabel(unsigned Id) const
{
  string Return = "";
  //TODO: if this operation is too slow then create a hash table
  for (map<string, unsigned>::const_iterator it = mLabelToId.begin(); it != mLabelToId.end(); ++it)
  {
    if (it->second == Id)
    {
      Return = it->first;
      break;
    }
  }

  return Return;
}

//=================================================================================================
//=================================================================================================
wxFileName CRecognitionDb::GetImageFileName(unsigned i) const
{
  return mImageFileNames[i];
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlSvmParams(CvSVMParams* pParams, ofstream& Os)
{
  // Output the support vector machine type
  switch (pParams->svm_type)
  {
    case CvSVM::C_SVC:     GenHtmlTableLine(Os, "<b>Type</b>", string("C_SVC"), 3);     break;
    case CvSVM::NU_SVC:    GenHtmlTableLine(Os, "<b>Type</b>", string("NU_SVC"), 3);    break;
    case CvSVM::ONE_CLASS: GenHtmlTableLine(Os, "<b>Type</b>", string("ONE_CLASS"), 3); break;
    case CvSVM::EPS_SVR:   GenHtmlTableLine(Os, "<b>Type</b>", string("EPS_SVR"), 3);   break;
    case CvSVM::NU_SVR:    GenHtmlTableLine(Os, "<b>Type</b>", string("NU_SVR"), 3);    break;
  }

  // Output the support vector machine kernel type
  switch (pParams->kernel_type)
  {
    case CvSVM::LINEAR:  GenHtmlTableLine(Os, "<b>Kernel Type</b>", string("LINEAR"), 3);  break;
    case CvSVM::POLY:    GenHtmlTableLine(Os, "<b>Kernel Type</b>", string("POLY"), 3);    break;
    case CvSVM::RBF:     GenHtmlTableLine(Os, "<b>Kernel Type</b>", string("RBF"), 3);     break;
    case CvSVM::SIGMOID: GenHtmlTableLine(Os, "<b>Kernel Type</b>", string("SIGMOID"), 3); break;
  }

  GenHtmlTableLine(Os, "<b>Degree (for POLY only)</b>", (unsigned)pParams->degree, 3);
  GenHtmlTableLine(Os, "<b>Coef0 (for POLY/SIGMOID)</b>", pParams->coef0, 3);
  GenHtmlTableLine(Os, "<b>Gamma (for POLY/RBF/SIGMOID)</b>", pParams->gamma, 3);

}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenSetupSummaryLog()
{
  wxFileName LogName = mDbDirs.mLogDir;
  LogName.SetName("~Summary.Setup");
  LogName.SetExt("html");

  ofstream Os(LogName.GetFullPath().c_str());

  if (!Os.is_open()) return;

  GenHtmlHeader(Os, "Summary Setup");

  wxDateTime CurrentTime = wxDateTime::Now();

  Os << "<h2>Database: " << mDbName << "</h2>\n";
  Os << "<b>Creation time: </b>" << CurrentTime.Format() << "\n";

  Os << "<h3>Top-level directories</h3>";
  Os << "<b>Database: </b>" << mTopLevelDirs.mDatabaseDir.GetFullPath() << "\n";
  Os << "<b>Image: </b>" << mTopLevelDirs.mImageDir.GetFullPath() << "\n";
  Os << "<b>Log: </b>" << mTopLevelDirs.mLogDir.GetFullPath() << "\n";
  Os << "<b>Setup: </b>" << mTopLevelDirs.mSetupDir.GetFullPath() << "\n";

  Os << "<h3>Database directories</h3>";
  Os << "<b>Database: </b>" << mDbDirs.mDatabaseDir.GetFullPath() << "<br>\n";
  Os << "<b>Image: </b>" << mDbDirs.mImageDir.GetFullPath() << "<br>\n";
  Os << "<b>Log: </b>" << mDbDirs.mLogDir.GetFullPath() << "<br>\n";
  Os << "<b>Setup: </b>" << mDbDirs.mSetupDir.GetFullPath() << "<br>\n";

  Os << "<h3>Entries</h3>\n";
  GenHtmlTableHeader(Os, 1, 3, 2);
  GenHtmlTableLine(Os, "<b>Entries</b>", (unsigned)mEntries.size(), 3);
  GenHtmlTableLine(Os, "<b>Classes</b>", (unsigned)mLabelToId.size(), 3);
  GenHtmlTableFooter(Os);

  // Example entry
  //<features type="SURF">
  //  <!-- Common parameters -->
  //  <octaves      value="4"/>
  //  <octaveLayers value="2"/>
  //  <threshold    value="4000"/>
  //  <!-- Dynamic adjuster parameters (SURF only) -->
  //  <adjusterOn   value="true"/>
  //  <adjusterMin  value="500"/>
  //  <adjusterMax  value="900"/>
  //  <adjusterIter value="50"/>
  //  <!-- Other parameters -->
  //  <autoLevels  value="false"/>
  //  <generateLog value="true"/>
  //  <cache       value="true"/>
  //</features>
  Os << "<h3>Feature parameters</h3>\n";
  GenHtmlTableHeader(Os, 1, 3, 2);
  GenHtmlTableLine(Os, "<b>Generate Feature Log</b>", mGenFeatureLog, 3);
  GenHtmlTableLine(Os, "<b>Perform image auto levels</b>", mAutoLevels, 3);
  GenHtmlTableLine(Os, "<b>Cache features</b>", mCacheFeatures, 3);
  GenHtmlTableFooter(Os);

  GenHtmlTableHeader(Os, 1, 3, 2);
  switch(mFeatureType)
  {
    case eSIFT:
      GenHtmlTableLine(Os, "<b>Feature Type</b>", "SIFT", 3);
      // Summarize common SIFT parameters
      if (mpSiftCommonParams)
      {
        GenHtmlTableLine(Os, "<b>Angle Mode</b>", (unsigned)(mpSiftCommonParams->angleMode), 3);
        GenHtmlTableLine(Os, "<b>Firsdt octave</b>", (unsigned)mpSiftCommonParams->firstOctave, 3);
        GenHtmlTableLine(Os, "<b>Octave layers</b>", (unsigned)mpSiftCommonParams->nOctaveLayers, 3);
        GenHtmlTableLine(Os, "<b>Octaves</b>", (unsigned)mpSiftCommonParams->nOctaves, 3);
      }
      else
      {
        GenHtmlTableLine(Os, "<b>ERROR:</b>", "No SIFT common parameters specified!", 3);
      }

      // Summary SIFT detector parameters
      if (mpSiftDetectorParams)
      {
        GenHtmlTableLine(Os, "<b>Edge Threshold</b>", mpSiftDetectorParams->edgeThreshold, 3);
        GenHtmlTableLine(Os, "<b>Threshold</b>", mpSiftDetectorParams->threshold, 3);
      }
      else
      {
        GenHtmlTableLine(Os, "<b>ERROR:</b>", "No SIFT detector parameters specified!", 3);
      }

      // Summarize SIFT descriptor parameters
      if (mpSiftDescriptorParams)
      {
        GenHtmlTableLine(Os, "<b>IsNormalize</b>", mpSiftDescriptorParams->isNormalize, 3);
        GenHtmlTableLine(Os, "<b>Recalcuate Angles</b>", mpSiftDescriptorParams->recalculateAngles, 3);
        GenHtmlTableLine(Os, "<b>Magnification</b>", mpSiftDescriptorParams->magnification, 3);
      }
      else
      {
        GenHtmlTableLine(Os, "<b>ERROR:</b>", "No SIFT descriptor parameters specified!", 3);
      }
    break;
    case eSURF:
      GenHtmlTableLine(Os, "<b>Feature Type</b>", "SURF", 3);
      if (mpSurfParams)
      {
        GenHtmlTableLine(Os, "<b>Octaves</b>",       (unsigned)(mpSurfParams->nOctaves), 3);
        GenHtmlTableLine(Os, "<b>Octave Layers</b>", (unsigned)(mpSurfParams->nOctaveLayers), 3);
        GenHtmlTableLine(Os, "<b>Threshold</b>",     mpSurfParams->hessianThreshold, 3);
        GenHtmlTableLine(Os, "<b>Extended</b>",      mSurfExtended, 3);
      }
      else
      {
        GenHtmlTableLine(Os, "<b>ERROR:</b>", "No SURF parameters specified!", 3);
      }

      // SURF only options
      GenHtmlTableLine(Os, "<b>Auto adjust threshold</b>", mAdjusterOn, 3);
      if (mAdjusterOn)
      {
        GenHtmlTableLine(Os, "<b>Auto adjust minimum features</b>", mAdjusterMin, 3);
        GenHtmlTableLine(Os, "<b>Auto adjust maximum features</b>", mAdjusterMax, 3);
        GenHtmlTableLine(Os, "<b>Auto adjust maximum iterations</b>", mAdjusterIter, 3);
        GenHtmlTableLine(Os, "<b>Auto adjust memory</b>", mAdjusterMemory, 3);
        GenHtmlTableLine(Os, "<b>Auto adjust learn rate</b>", mAdjusterLearnRate, 3);
      }
    break;
  }
  GenHtmlTableFooter(Os);

  // Histogram generation example
  //<histograms type="color">
  //  <bins       value="64"/>
  //  <log        value="true"/>
  //</histograms>
  Os << "<h3>Color Histogram Parameters</h3>\n";
  GenHtmlTableHeader(Os, 1, 3, 2);
  GenHtmlTableLine(Os, "<b>Bins</b>", mColorHistogramBins, 3);
  GenHtmlTableLine(Os, "<b>Log</b>", mGenColorHistogramLog, 3);
  GenHtmlTableFooter(Os);

  // Color classifier example
  //<classifier type="svm" input="color">
  //  <type       value="C_SVC"/>
  //  <kernelType value="POLY"/>
  //  <gamma      value="0.5"/>
  //  <degree     value="3"/>
  //  <log        value="true"/>
  //</classifier>
  Os << "<h3>Color histogram classifier parameters</h3>\n";
  GenHtmlTableHeader(Os, 1, 3, 2);
  GenHtmlSvmParams(mpColorClassifierParams, Os);
  GenHtmlTableLine(Os, "<b>Log</b>", mGenColorClassifierLog, 3);
  GenHtmlTableFooter(Os);

  // Dictionary generation example
  //<dictionary type="kmeans">
  //  <iterations value="50"/>
  //  <words      value="22"/>
  //  <log        value="true"/>
  //  <cache      value="true"/>
  //</dictionary>
  Os << "<h3>Dictionary parameters</h3>\n";
  GenHtmlTableHeader(Os, 1, 3, 2);
  GenHtmlTableLine(Os, "<b>Words</b>", mWordCount, 3);
  GenHtmlTableLine(Os, "<b>K-means iterations</b>", mWordKMeansIter, 3);
  GenHtmlTableLine(Os, "<b>Cache dictionary</b>", mCacheDictionary, 3);
  GenHtmlTableLine(Os, "<b>Log</b>", mGenColorClassifierLog, 3);
  GenHtmlTableFooter(Os);

  // Word classifier example
  //<classifier type="svm" input="words">
  //  <type       value="C_SVC"/>
  //  <kernelType value="POLY"/>
  //  <gamma      value="0.5"/>
  //  <degree     value="3"/>
  //  <log        value="true"/>
  //</classifier>
  Os << "<h3>Word classifier parameters</h3>\n";
  GenHtmlTableHeader(Os, 1, 3, 2);
  GenHtmlSvmParams(mpWordClassifierParams, Os);

  GenHtmlTableLine(Os, "<b>Log</b>", mGenColorHistogramLog, 3);
  GenHtmlTableFooter(Os);

  GenHtmlFooter(Os);

  Os.close();
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenEntrySummaryLog(std::string HtmlPath)
{
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenColorHistLogHtml()
{
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenFeatureLogImage(
  const Mat& Image,
  const CRecognitionEntry& Entry)
{
  const int flags = DrawMatchesFlags::DRAW_OVER_OUTIMG|DrawMatchesFlags::DRAW_RICH_KEYPOINTS;

  Mat LogImage(Image);
  drawKeypoints(Image, Entry.GetKeyPoints(), LogImage, Scalar(1.0, 1.0, 1.0), flags);

  wxFileName LogImageName = mDbDirs.mLogDir;
  LogImageName.SetName(Entry.GetName() + ".Features");
  LogImageName.SetExt("png");

  cv::imwrite(LogImageName.GetFullPath().ToStdString(), LogImage);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenFeatureLogHtml(const CRecognitionEntry& Entry, wxTimeSpan& GenTime)
{
  wxString Name = wxString(Entry.GetName().c_str());

  wxFileName LogImageName = mDbDirs.mLogDir;
  LogImageName.SetName(Name + ".Features");
  LogImageName.SetExt("png");

  wxFileName LogName = mDbDirs.mLogDir;
  LogName.SetName(Name + ".Features");
  LogName.SetExt("html");

  wxFileName OrignialImageName = mDbDirs.mImageDir;
  OrignialImageName.SetName(Name);
  OrignialImageName.SetExt("jpg");

  ofstream Os(LogName.GetFullPath().c_str());

  if (!Os.is_open()) return;

  GenHtmlHeader(Os, Entry.GetName());

  // Images
  GenHtmlTableHeader(Os, 100, 1, 2);
  string ImageA = "<img src='../../" + OrignialImageName.GetFullPath().ToStdString() + "'>";
  string ImageB = "<img src='" + LogImageName.GetFullName().ToStdString() + "'>";
  GenHtmlTableLine(Os, ImageA, ImageB, 3);
  GenHtmlTableLine(Os, "<b>Original</b>" , "<b>Keypoints</b>", 3);

  // Entry properties
  GenHtmlTableHeader(Os, 1, 1, 2);
  GenHtmlTableLine(Os, "<b>Name</b>" ,     Entry.GetName() , 3);
  GenHtmlTableLine(Os, "<b>Id</b>",        Entry.GetLabelId(), 3);
  GenHtmlTableLine(Os, "<b>Keypoints</b>", Entry.GetKeyPointCount(), 3);

  // Print the size that only the descriptors take up
  const unsigned DescriptorLength = Entry.GetDescriptors().cols;
  unsigned SizeInBytes = Entry.GetKeyPointCount()*DescriptorLength*sizeof(float);
  GenHtmlTableLine(Os, "<b>Size (Bytes)</b>", SizeInBytes, 3);

  string GenTimeString = GenTime.Format("%M:%S:%l").ToStdString();
  GenHtmlTableLine(Os, "<b>Generation time</b>", GenTimeString , 3);

  GenHtmlTableFooter(Os);
  GenHtmlFooter(Os);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenWordLogHtml()
{

  for (unsigned i = 0; i < mEntries.size(); i++)
  {
    CRecognitionEntry& Entry = mEntries.at(i);
    wxString Name = wxString(Entry.GetName().c_str());

    wxFileName LogImageName = mDbDirs.mLogDir;
    LogImageName.SetName(Name + ".Words");
    LogImageName.SetExt("png");

    wxFileName LogName = mDbDirs.mLogDir;
    LogName.SetName(Name + ".Words");
    LogName.SetExt("html");

    WriteHistogramImage(LogImageName, Entry.GetWordHist());

    ofstream Os(LogName.GetFullPath().c_str());
    if (!Os.is_open()) return;

    GenHtmlHeader(Os, Entry.GetName());

    // Images
    GenHtmlTableHeader(Os, 1, 1, 2);

    string LogImageHtml = "<img src='" + LogImageName.GetFullName().ToStdString() + "'>";
    GenHtmlTableLine(Os, LogImageHtml, "", 3);
    GenHtmlTableLine(Os, "<b>Word Histogram</b>" , "", 3);

    // Entry properties
    GenHtmlTableHeader(Os, 1, 1, 2);

    GenHtmlTableLine(Os, "<b>Name</b>" , Entry.GetName() , 3);
    GenHtmlTableLine(Os, "<b>Id</b>",    Entry.GetLabelId(), 3);
    GenHtmlTableLine(Os, "<b>Words</b>", Entry.GetKeyPointCount(), 3);

    const Mat& WordHist = Entry.GetWordHist();

    for (unsigned i = 0; i < (unsigned)WordHist.cols; i++)
    {
      wxString Name = wxString::Format("<b>%d</b>", i);
      GenHtmlTableLine(Os, Name.ToStdString(), (unsigned)WordHist.at<int>(0,i), 3);
    }

    GenHtmlTableFooter(Os);
    GenHtmlFooter(Os);
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlHeader(ostream& Os, string Title)
{
  Os << "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01//EN\"";
  Os << "\"http://www.w3.org/TR/html4/strict.dtd\">";
  Os << "\n<HTML>\n  <HEAD>\n    <TITLE>" << Title << "</TITLE>\n  </HEAD>\n  <BODY>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlFooter(ostream& Os)
{
  Os << "  </BODY>\n</HTML>";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableHeader(
  ostream& Os,
  unsigned Padding,
  unsigned Spacing,
  unsigned Indent)
{
  GenHtmlIndent(Os, Indent);
  Os << "<TABLE cellpadding='" << Padding << "' cellspacing='" << Spacing << "'>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableFooter(ostream& Os, unsigned Indent)
{
  GenHtmlIndent(Os, Indent);
  Os << "</TABLE>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableRowBeg(ostream& Os, unsigned Indent)
{
  GenHtmlIndent(Os, Indent);
  Os << "<TR>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableRowEnd(ostream& Os, unsigned Indent)
{
  GenHtmlIndent(Os, Indent);
  Os << "</TR>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableDivBeg(ostream& Os, unsigned Indent)
{
  GenHtmlIndent(Os, Indent);
  Os << "<TD>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableDivEnd(ostream& Os, unsigned Indent)
{
  GenHtmlIndent(Os, Indent);
  Os << "</TD>\n";
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlIndent(ostream& Os, unsigned Indent)
{
  for (unsigned i = 0; i < Indent; i++)
  {
    Os << "  ";
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableLine(ostream& Os, const vector<string>& Values, unsigned Indent)
{
  GenHtmlTableRowBeg(Os, Indent);

  for (unsigned i = 0; i < Values.size(); i++)
  {
    GenHtmlTableDivBeg(Os, Indent+1);
    GenHtmlIndent(Os, Indent+2);
    Os << Values[i] << "\n";
    GenHtmlTableDivEnd(Os, Indent+1);
  }

  GenHtmlTableRowEnd(Os, Indent);
}


//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableLine(ostream& Os, string Name , string Value , unsigned Indent)
{
  GenHtmlTableRowBeg(Os, Indent);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  Os << Name << "\n";
  GenHtmlTableDivEnd(Os, Indent+1);

  if (Value.length() != 0)
  {
    GenHtmlTableDivBeg(Os, Indent+1);
    GenHtmlIndent(Os, Indent+2);
    Os << Value << "\n";
    GenHtmlTableDivEnd(Os, Indent+1);
  }
  GenHtmlTableRowEnd(Os, Indent);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableLine(ostream& Os, string Name , unsigned Value , unsigned Indent)
{
  GenHtmlTableRowBeg(Os, Indent);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  Os << Name << "\n";
  GenHtmlTableDivEnd(Os, Indent+1);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  Os << Value << "\n";
  GenHtmlTableDivEnd(Os, Indent+1);
  GenHtmlTableRowEnd(Os, Indent);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableLine(ostream& Os, string Name , double Value , unsigned Indent)
{
  GenHtmlTableRowBeg(Os, Indent);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  Os << Name << "\n";
  GenHtmlTableDivEnd(Os, Indent+1);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  Os << Value << "\n";
  GenHtmlTableDivEnd(Os, Indent+1);
  GenHtmlTableRowEnd(Os, Indent);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::GenHtmlTableLine(ostream& Os, string Name , bool Value , unsigned Indent)
{
  GenHtmlTableRowBeg(Os, Indent);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  Os << Name << "\n";
  GenHtmlTableDivEnd(Os, Indent+1);
  GenHtmlTableDivBeg(Os, Indent+1);
  GenHtmlIndent(Os, Indent+2);
  if (Value)
  {
    Os << "True\n";
  }
  else
  {
    Os << "False\n";
  }
  GenHtmlTableDivEnd(Os, Indent+1);
  GenHtmlTableRowEnd(Os, Indent);
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadHistogramsElement(TiXmlElement* pHistograms)
{
  string DictionaryType;

  DictionaryType = ReadTypeAttribute(pHistograms);

  // Right now this is the only method supported
  if (DictionaryType == "color")
  {
    // Set to the defaults in case there is a read failure
    int ColorHistogramBins = 256;

    for (
      TiXmlElement* pElement = pHistograms->FirstChildElement();
      pElement != 0;
      pElement = pElement->NextSiblingElement())
    {

      string Param = pElement->Value();

      if (Param == "bins")
      {
        ReadIntValueAttribute(pElement, &ColorHistogramBins);
        if ((ColorHistogramBins > 0) && (ColorHistogramBins <= 256))
        {
          mColorHistogramBins = ColorHistogramBins;
        }
        else
        {
          cout << "WARNING: Number of histogram bins is invalid, using default value\n";
        }
      }
      else if (Param == "log")
      {
        ReadBoolValueAttribute(pElement, &mGenColorHistogramLog);
      }
      else if (Param == "cache")
      {
        //ReadBoolValueAttribute(pElement, &mCacheDictionary);
      }
    }
  }
  else
  {
    cout << "WARNING: unsupported histogram type\n";
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadDictionaryElement(TiXmlElement* pDictionary)
{
  string DictionaryType;

  DictionaryType = ReadTypeAttribute(pDictionary);

  // Right now this is the only method supported
  if (DictionaryType == "kmeans")
  {
    // Set to the defaults in case there is a read failure
    int Words = mWordCount;
    int Iterations = mWordKMeansIter;

    for (
      TiXmlElement* pElement = pDictionary->FirstChildElement();
      pElement != 0;
      pElement = pElement->NextSiblingElement())
    {

      string Param = pElement->Value();

      if (Param == "iterations")
      {
        ReadIntValueAttribute(pElement, &Iterations);
        if ((Iterations > 0) && (Iterations <= 1000000))
        {
          mWordKMeansIter = Iterations;
        }
        else
        {
          cout << "WARNING: Iteration count is invalid, using default value\n";
        }
      }
      else if (Param == "words")
      {
        ReadIntValueAttribute(pElement, &Words);
        if ((Words > 0) && (Words <= 1000000))
        {
          mWordCount = Words;
        }
        else
        {
          cout << "WARNING: Word count is invalid, using default value\n";
        }
      }
      else if (Param == "log")
      {
        ReadBoolValueAttribute(pElement, &mGenWordLog);
      }
      else if (Param == "cache")
      {
        ReadBoolValueAttribute(pElement, &mCacheDictionary);
      }
    }
  }
  else
  {
    cout << "WARNING: Unsupported dictionary type\n";
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadClassifierElement(TiXmlElement* pClassifier)
{
  string ClassifierType = ReadTypeAttribute(pClassifier);
  string ClassifierInput = ReadInputAttribute(pClassifier);

  // Right now this is the only method supported
  if (ClassifierType == "svm")
  {
    CvSVMParams* Params = new CvSVMParams();
    Params->degree = 3;
    Params->gamma = 0.5;
    Params->kernel_type = CvSVM::POLY;
    Params->svm_type = CvSVM::C_SVC;

    bool GenLog = false;

    for (
      TiXmlElement* pElement = pClassifier->FirstChildElement();
      pElement != 0;
      pElement = pElement->NextSiblingElement())
    {

      string Param = pElement->Value();

      if (Param == "type")
      {
        string Type = ReadValueAttribute(pElement);

        if (Type == "C_SVC")
        {
          Params->svm_type = CvSVM::C_SVC;
        }
        else
        {
          cout << "ERROR: " << Type << " is not a supported classifier type\n";
        }
      }
      else if (Param == "kernelType")
      {
        string Type = ReadValueAttribute(pElement);

        if (Type == "LINEAR")
        {
          Params->kernel_type = CvSVM::LINEAR;
        }
        else if (Type == "POLY")
        {
          Params->kernel_type = CvSVM::POLY;
        }
        else if (Type == "RBF")
        {
          Params->kernel_type = CvSVM::RBF;
        }
        else if (Type == "SIGMOID")
        {
          Params->kernel_type = CvSVM::SIGMOID;
        }
      }
      else if (Param == "gamma")
      {
        // Fill with default value in case there is a read error
        double Gamma = Params->gamma;

        ReadDoubleValueAttribute(pElement, &Gamma);

        if ((Gamma > 0) && (Gamma < 10))
        {
          Params->degree = Gamma;
        }
      }
      else if (Param == "degree")
      {
        // Fill with default value in case there is a read error
        int Degree = Params->degree;

        ReadIntValueAttribute(pElement, &Degree);

        // Make sure value is in valid range
        if ((Degree > 0) && (Degree < 20))
        {
          Params->degree = Degree;
        }
      }
      else if (Param == "log")
      {
        ReadBoolValueAttribute(pElement, &GenLog);
      }
    }

    if (ClassifierInput == "words")
    {
      mGenWordClassifierLog = GenLog;
      mpWordClassifierParams = Params;
    }
    else if (ClassifierInput == "color")
    {
      mGenColorClassifierLog = GenLog;
      mpColorClassifierParams = Params;
    }
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadEntryElement(TiXmlElement* pEntry)
{

  wxFileName ImagePath = mDbDirs.mImageDir;
  string Label;
  string Comment;

  for (
    TiXmlAttribute* pAttrib = pEntry->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "label")
    {
      Label = pAttrib->Value();
    }
    else if (Name == "file")
    {
      ImagePath.SetFullName(pAttrib->Value());
    }
    else if (Name == "comment")
    {
      Comment = pAttrib->Value();
    }
  }

  if (ImagePath.IsFileReadable() && Label.size())
  {
    mImageFileNames.push_back(ImagePath);

    map<string,unsigned>::iterator it;

    it = mLabelToId.find(Label);

    unsigned Id;
    if (it == mLabelToId.end())
    {
      Id = mLabelToId.size();
      mLabelToId[Label] = Id;
    }
    else
    {
      Id = it->second;
    }
    mEntries.push_back(CRecognitionEntry(ImagePath.GetName().ToStdString(), Id));
  }
  else
  {
    cout << "WARNING: " << ImagePath.GetFullPath() << " could not be found\n";
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadDisplayElement(TiXmlElement* pEntry)
{
  for (
    TiXmlElement* pElement = pEntry->FirstChildElement();
    pElement != 0;
    pElement = pElement->NextSiblingElement())
  {

    string Param = pElement->Value();

    if (Param == "color")
    {
      int R = 255;
      int G = 255;
      int B = 255;
      for (
        TiXmlAttribute* pAttrib = pElement->FirstAttribute();
        pAttrib != 0;
        pAttrib = pAttrib->Next())
      {
        string Name = pAttrib->Name();

        if (Name == "r")
        {
          pAttrib->QueryIntValue(&R);
        }
        if (Name == "g")
        {
          pAttrib->QueryIntValue(&G);
        }
        if (Name == "b")
        {
          pAttrib->QueryIntValue(&B);
        }
      }
      mLabelColors.push_back(Scalar(B,G,R));
    }
  }
}

//=================================================================================================
//=================================================================================================
string CRecognitionDb::ReadTypeAttribute(TiXmlElement* pElement)
{
  for (
    TiXmlAttribute* pAttrib = pElement->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "type")
    {
      return string(pAttrib->Value());
      break;
    }
  }
  return string("");
}

//=================================================================================================
//=================================================================================================
string CRecognitionDb::ReadInputAttribute(TiXmlElement* pElement)
{
  for (
    TiXmlAttribute* pAttrib = pElement->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "input")
    {
      return string(pAttrib->Value());
      break;
    }
  }
  return string("");
}

//=================================================================================================
//=================================================================================================
string CRecognitionDb::ReadValueAttribute(TiXmlElement* pElement)
{
  for (
    TiXmlAttribute* pAttrib = pElement->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "value")
    {
      return string(pAttrib->Value());
      break;
    }
  }
  return string("");
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadIntValueAttribute(TiXmlElement* pElement, int* pVal)
{
  for (
    TiXmlAttribute* pAttrib = pElement->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "value")
    {
      if (pAttrib->QueryIntValue(pVal)== TiXmlAttribute::TIXML_NO_ERROR)
      {
        return;
      }
    }
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadDoubleValueAttribute(TiXmlElement* pElement, double* pVal)
{
  for (
    TiXmlAttribute* pAttrib = pElement->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "value")
    {
      if (pAttrib->QueryDoubleValue(pVal)== TiXmlAttribute::TIXML_NO_ERROR)
      {
        return;
      }
    }
  }
}

//=================================================================================================
//=================================================================================================
void CRecognitionDb::ReadBoolValueAttribute(TiXmlElement* pElement, bool* pVal)
{
  for (
    TiXmlAttribute* pAttrib = pElement->FirstAttribute();
    pAttrib != 0;
    pAttrib = pAttrib->Next())
  {
    string Name = pAttrib->Name();

    if (Name == "value")
    {
      string BoolVal = pAttrib->Value();

      if (BoolVal == "true")
      {
        *pVal = true;
      }
      else
      {
        *pVal = false;
      }
    }
  }
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::LoadDictionary(ifstream& Is)
{

  int Rows = 0;
  int Cols = 0;

  Is.read(reinterpret_cast<char*>(&Rows), sizeof(Rows));
  Is.read(reinterpret_cast<char*>(&Cols), sizeof(Cols));

  if ((Rows != mWordCount) || (Cols <= 0)) return false;

  if (mpDictionary != 0) delete(mpDictionary);

  mpDictionary = new Mat(Rows, Cols, CV_32F, Scalar(0));

  for (unsigned i=0; i < (unsigned)Rows; i++)
  {
    for (unsigned j = 0; j < (unsigned)Cols; j++)
    {
      float Val;
      Is.read(reinterpret_cast<char*>(&(Val)), sizeof(Val));
      mpDictionary->at<float>(i,j) = Val;
    }
  }

  return true;
}

//=================================================================================================
//=================================================================================================
bool CRecognitionDb::SaveDictionary(ofstream& Os)
{
  if (mpDictionary == 0) return false;

  int& Rows = mpDictionary->rows;
  int& Cols = mpDictionary->cols;

  Os.write(reinterpret_cast<char*>(&Rows), sizeof(Rows));
  Os.write(reinterpret_cast<char*>(&Cols), sizeof(Cols));

  for (unsigned i = 0; i < (unsigned)Rows; i++)
  {
    for (unsigned j = 0; j < (unsigned)Cols; j++)
    {
      float Val = mpDictionary->at<float>(i,j);
      Os.write(reinterpret_cast<char*>(&(Val)), sizeof(Val));
    }
  }

  return true;
}