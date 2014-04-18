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
// STL
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>

// Visual Recogntion Lib
#include <RecognitionDb.h>

// wxWidgets
#include <wx/wx.h>
#include <wx/socket.h>
#include <wx/string.h>
#include <wx/filename.h>
#include <wx/textctrl.h>
#include <wx/timer.h>

// OpenCV
#include <highgui.h>

using namespace std;

//=================================================================================================
// Output a summary log 
//=================================================================================================
void GenTimingSummaryCsv(
  ofstream& Os,
  const CRecognitionDb& TrainDb,
  const CRecognitionDb& TestDb,
  wxTimeSpan FeaturesTime,    // Time it takes to generate features
  wxTimeSpan DictionaryTime,  // Time it takes to create a dictionary of visual words
  wxTimeSpan WordHistTime,    // Time it takes to generate the bag of visual word histograms
  wxTimeSpan TrainWordTime,   // Time it takes to train the bag of visual words classifier
  wxTimeSpan TrainColorTime,  // Time it takes to train the color classifier
  wxTimeSpan ColorHistTime,   // Time it takes to generate the color histograms
  wxTimeSpan WordVerifyTime,  // Time it takes to perform verification using visual words
  wxTimeSpan ColorVerifyTime, // Time it takes to perform verification using color
  bool WriteHeader)
{

  if (WriteHeader)
  {
    Os << "Test Database Name, Train Database Name, Feature Creation Time (ms),";
    Os << "Dictionary Creation Time (ms), BOVW Histogram Creation Time (ms),";
    Os << "BOVW Classifier Training Time (ms), Color Classifier Training Time (ms),";
    Os << "Color Histogram Creation Time (ms), Word Verification Time (ms),";
    Os << "Color Verification Time (ms),";
    Os << "Number of Train Db Entries, Number of Test Db Entries\n";
  }

  Os << TestDb.GetName()                               << ",";
  Os << TrainDb.GetName()                              << ",";
  Os << FeaturesTime.GetMilliseconds().ToString()      << ",";
  Os << DictionaryTime.GetMilliseconds().ToString()    << ",";
  Os << WordHistTime.GetMilliseconds().ToString()      << ",";
  Os << TrainWordTime.GetMilliseconds().ToString()     << ",";
  Os << TrainColorTime.GetMilliseconds().ToString()    << ",";
  Os << ColorHistTime.GetMilliseconds().ToString()     << ",";
  Os << WordVerifyTime.GetMilliseconds().ToString()    << ",";
  Os << ColorVerifyTime.GetMilliseconds().ToString()   << ",";
  Os << TrainDb.GetEntryCount()                        << ",";
  Os << TestDb.GetEntryCount()                         << "\n";
}

//=================================================================================================
// Train three databases using different visual word counts and verify each classifier on a test
// database
//=================================================================================================
void WordTest()
{
  cout << "=========================WORD TEST==========================\n";

  vector<string> WordTrainSetupFiles;
  WordTrainSetupFiles.push_back("HomogeneousLittleDogSmallTrain.Words05.xml");
  WordTrainSetupFiles.push_back("HomogeneousLittleDogSmallTrain.Words10.xml");
  WordTrainSetupFiles.push_back("HomogeneousLittleDogSmallTrain.Words15.xml");
  WordTrainSetupFiles.push_back("HomogeneousLittleDogSmallTrain.Words20.xml");
  WordTrainSetupFiles.push_back("HomogeneousLittleDogSmallTrain.Words25.xml");
  WordTrainSetupFiles.push_back("HomogeneousLittleDogSmallTrain.Words30.xml");

  ofstream VerifySummaryOs;
  ofstream TimingSummaryOs;
  string VerifySummaryLogName = "WordTest.Verify.csv";
  string TimingSummaryLogName = "WordTest.Timing.csv";

  VerifySummaryOs.open(VerifySummaryLogName.c_str());
  TimingSummaryOs.open(TimingSummaryLogName.c_str());

  struct CRecognitionDb::SDirs Dirs;
  Dirs.mDatabaseDir = wxFileName("database/");
  Dirs.mImageDir    = wxFileName("images/");
  Dirs.mLogDir      = wxFileName("logs/");
  Dirs.mSetupDir    = wxFileName("setup/");

 //Prepare the test dataset (only one)
  CRecognitionDb TestDb;
  TestDb.OnInit(Dirs, "HomogeneousLittleDogSmallTest.xml");
  TestDb.PopulateFeatures();

  bool WriteHeader = true;

  // For each input setup file
  for (int i = 0; i < WordTrainSetupFiles.size(); i++)
  {
    if (i > 0) WriteHeader = false;
    CRecognitionDb TrainDb;
    TrainDb.OnInit(Dirs, WordTrainSetupFiles[i]);

    // Time variables
    wxTimeSpan FeaturesTime(0);
    wxTimeSpan DictionaryTime(0);
    wxTimeSpan WordHistTime(0);
    wxTimeSpan TrainWordTime(0);
    wxTimeSpan TrainColorTime(0);
    wxTimeSpan ColorHistTime(0);
    wxTimeSpan WordVerifyTime(0);
    wxTimeSpan ColorVerifyTime(0);

    cout << "DATABASE: " << WordTrainSetupFiles[i] << " Started\n";

    cout << "    Populating features...........";
    TrainDb.PopulateFeatures(FeaturesTime);
    cout << "Time: " << FeaturesTime.Format("%M:%S:%l") << "\n";

    cout << "    Populating dictionary.........";
    TrainDb.PopulateDictionary(DictionaryTime, WordHistTime);
    cout << "Time: " << DictionaryTime.Format("%M:%S:%l") << " ";
    cout << WordHistTime.Format("%M:%S:%l") << "\n";

    cout << "    Training word classifier......";
    TrainDb.TrainWordClassifier(TrainWordTime);
    cout << "Time: " << TrainWordTime.Format("%M:%S:%l") << "\n";

    // Classification result variables
    vector<string> Classify;
    vector<string> Truth;
    map<string, unsigned> MatchCount;
    map<string, unsigned> TruthCount;

    // Self verification (test the trained classifier against itself)
    TrainDb.ClassifyDbWords(TrainDb, Classify, Truth, MatchCount, TruthCount, TrainWordTime, true);

    // Test verification (test the trained classifier against new images)
    TrainDb.ClassifyDbWords(TestDb, Classify, Truth, MatchCount, TruthCount, WordVerifyTime, true);
    TrainDb.GenClassifyDbSummaryCsv(
      VerifySummaryOs, WordTrainSetupFiles[i], MatchCount, TruthCount, WriteHeader);

    // Generate a summary for all of the tests
    GenTimingSummaryCsv(
      TimingSummaryOs, TrainDb, TestDb,
      FeaturesTime,
      DictionaryTime,
      WordHistTime,
      TrainWordTime,
      TrainColorTime,
      ColorHistTime,
      WordVerifyTime,
      ColorVerifyTime, WriteHeader);
    cout << "DATABASE: " << WordTrainSetupFiles[i] << " Finished!\n";
  }
  // Clean up
  TimingSummaryOs.close();
  VerifySummaryOs.close();

  cout << "============================================================\n";
}

//=================================================================================================
//=================================================================================================
int main(void)
{
  WordTest();

  // Used to stop windows console from closing
  int SomeUserInput;
  cin >> SomeUserInput;
  return 0;
}
