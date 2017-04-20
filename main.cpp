#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <fstream>
#include <io.h>
#include <vector>
using namespace std;
using namespace cv;


//--------BOW单词数
int wordCount = 2000; 
//--------BOW分词
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
BOWKMeansTrainer bowTrainer(wordCount, tc, 1, KMEANS_PP_CENTERS);
//--------BOW特征提取
Ptr<DescriptorExtractor> siftextractor = DescriptorExtractor::create("SIFT");
Ptr<DescriptorExtractor> surfextractor = DescriptorExtractor::create("SURF");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
//BOWImgDescriptorExtractor bowDesExtrac(surfextractor, matcher);
BOWImgDescriptorExtractor bowDesExtrac(siftextractor, matcher);
//--------SURF特征提取
SurfFeatureDetector surfdetector(500);
SurfFeatureDetector siftdetector(500);
//--------image读取位置
const string RootPath = "F://Current/MachineLearning/imagedata";
const string TrainPath = "C://Users/Lenovo/Desktop/images/training";
const string TestPath = "C://Users/Lenovo/Desktop/images/testing";
string ClassPath[19] = { "/car", "/city", "/dog", "/earth", "/fireworks", "/flowers", "/fruits",
"/glass", "/gold", "/gun", "/goldenfish", "/shoe", "/teapot", "/bear", "/dragonfly", "/football", "/plane", "/sky", "/worldcup" };


int main(){

	//--------读取所有图片.得到bowTrainer：所有surf特征的合集
	cout << "extracting surf feature from the pictures....." << endl;
	for (int i = 0; i < 19; i++){
		String TmpPath = TrainPath + ClassPath[i];
		long hFile = 0;
		struct _finddata_t fileInfo;
		string pathName, exdName;
		if ((hFile = _findfirst(pathName.assign(TmpPath).append("\\*").c_str(), &fileInfo)) == -1) {
			return 0;
		}
		do {
			if (fileInfo.name[0] != ClassPath[i][1]) continue;
			Mat tmp_image = imread(TmpPath + "/" + fileInfo.name);
			cout << fileInfo.name << endl;
			vector<KeyPoint> tmp_keypoint;
			//用探测器探测到SIFT，但keypoint只是点的位置，还没有处理成向量
			siftdetector.detect(tmp_image, tmp_keypoint);
			//surfdetector.detect(tmp_image, tmp_keypoint);
			//将对应的特征内容写入mat
			Mat tmp_feature;
			siftextractor->compute(tmp_image, tmp_keypoint, tmp_feature);
			//surfextractor->compute(tmp_image, tmp_keypoint, tmp_feature);
			if (tmp_feature.empty()) continue;
			bowTrainer.add(tmp_feature);
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}

	//--------对surf特征进行聚类
	cout << "Clustering the features" << endl;
	Mat vocabulary = bowTrainer.cluster();
	bowDesExtrac.setVocabulary(vocabulary);
	cout << "Get the vocabulary!" << endl;

	//--------使用单词本计算新的词频,得到向量，准备好SVM的训练集training data和labels
	cout << "extracting histograms in the form of BOW for each image " << endl;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, wordCount, CV_32FC1);
	for (int i = 0; i < 19; i++){
		long hFile = 0;
		string pathName, exdName;
		struct _finddata_t fileInfo;
		//String tmpname = to_string(i);
		String TmpPath = TrainPath + ClassPath[i];
		if ((hFile = _findfirst(pathName.assign(TmpPath).append("\\*").c_str(), &fileInfo)) == -1) {
			return 0;
		}
		do {
			if (fileInfo.name[0] != ClassPath[i][1]) continue;
			Mat tmp_image = imread(TmpPath + "/" + fileInfo.name);
			cout << fileInfo.name << endl;
			vector<KeyPoint> tmp_keypoint;
			siftdetector.detect(tmp_image, tmp_keypoint);
			//surfdetector.detect(tmp_image, tmp_keypoint);
			Mat tmp_bowdescriptor;
			bowDesExtrac.compute(tmp_image, tmp_keypoint, tmp_bowdescriptor);//tmp_descriptor就是每张图的bow直方图表示
			if (tmp_bowdescriptor.empty()) continue;
			trainingData.push_back(tmp_bowdescriptor);
			labels.push_back((float)i);
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}

	//--------SVM训练
	//设置参数
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
	CvSVM svm;

	cout << "training the SVM" << endl;
	bool res = svm.train(trainingData, labels, Mat(), Mat(), params);
	cout << res << endl;
	cout << "Processing evaluation data..." << endl;

	
	
	
	//--------SVM测试
	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, wordCount, CV_32FC1);
	int k = 0;
	Mat results(0, 1, CV_32FC1);
	for (int i = 0; i < 19; i++){
		long hFile = 0;
		string pathName, exdName;
		struct _finddata_t fileInfo;
		//String tmpname = to_string(i);
		String TmpPath = TestPath + ClassPath[i];
		if ((hFile = _findfirst(pathName.assign(TmpPath).append("\\*").c_str(), &fileInfo)) == -1) {
			return 0;
		}
		do {
			if (fileInfo.name[0] != ClassPath[i][1]) continue;
			Mat tmp_image = imread(TmpPath + "/" + fileInfo.name);
			vector<KeyPoint> tmp_keypoint;
			siftdetector.detect(tmp_image, tmp_keypoint);
			//surfdetector.detect(tmp_image, tmp_keypoint);
			Mat tmp_bowdescriptor;
			bowDesExtrac.compute(tmp_image, tmp_keypoint, tmp_bowdescriptor);//tmp_descriptor就是每张图的bow直方图表示
			evalData.push_back(tmp_bowdescriptor);
			groundTruth.push_back((float)i);
			float response = svm.predict(tmp_bowdescriptor);
			results.push_back(response);
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}


	//calculate the number of unmatched classes 
	double errorRate = (double)countNonZero(groundTruth - results) / evalData.rows;
	cout << "Error rate is " << errorRate << endl;

	return 0;
}
