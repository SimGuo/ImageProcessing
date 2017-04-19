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

//const string RootPath = "C://Users/Lenovo/Desktop/images/training";  //简化的训练集
const string RootPath = "F://Current/MachineLearning/imagedata";
/*
double PresetLabels[5][19] = {
	{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 },
	{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0 },
	{ 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0 },
	{ 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 },
	{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 } };
  */
string ClassPath[19] = { "/car", "/city", "/dog", "/earth", "/fireworks", "/flowers", "/fruits",
"/glass", "/gold", "/gun", "/goldenfish", "/shoe", "/teapot", "/bear", "/dragonfly", "/football", "/plane", "/sky", "/worldcup" };
int wordCount = 30; //生成多少个单词
/*
*	SIFT、SURF特征探测器
*	SIFT特征和SURF提取成向量的结构
*	allDescriptors中是所有的图片对应的sift特征的并集
*/
SiftFeatureDetector siftdtc;
SurfFeatureDetector surfdtc;
SiftDescriptorExtractor siftextract;
SurfDescriptorExtractor surfextract;


Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
BOWImgDescriptorExtractor bowextract(extractor, matcher);


int WriteData(string fileName, cv::Mat& matData) {
	if (matData.empty()){
		cout << "矩阵为空" << endl;
		return 1;
	}

	ofstream outFile(fileName.c_str(), ios_base::out);
	if (!outFile.is_open()){
		cout << "打开文件失败" << endl;
		return -1;
	}

	for (int r = 0; r < matData.rows; r++){
		for (int c = 0; c < matData.cols; c++){
			int data = matData.at<int>(r, c); 
			outFile << data << ",";
		}
		outFile << endl;  //换行  
	}

	return 0;
}
int extractfeature_to_csv(){

	/*
	*	提取图片的SIFT特征到txt文件的完整流程
	*	此处SIFT未经过筛选
	*
	*/
	Mat allDescriptors;
	for (int i = 0; i < 19; i++){
		String TmpPath = RootPath + ClassPath[i];
		long hFile = 0;
		struct _finddata_t fileInfo;
		string pathName, exdName;
		if ((hFile = _findfirst(pathName.assign(TmpPath).append("\\*").c_str(), &fileInfo)) == -1) {
			return 0;
		}
		do {
			if (fileInfo.name[0] != ClassPath[i][1]) continue;
			Mat tmp_image = imread(TmpPath + "/" + fileInfo.name);
			printf("%s\n", fileInfo.name);
			vector<KeyPoint> tmp_keypoint;
			//用探测器探测到SIFT，但keypoint只是点的位置，还没有处理成向量
			siftdtc.detect(tmp_image, tmp_keypoint);					
			//surfdtc.detect(tmp_image, tmp_keypoint);
			//将对应的特征内容写入mat
			Mat tmp_descriptor;
			siftextract.compute(tmp_image, tmp_keypoint, tmp_descriptor);
			//surfextract.compute(tmp_image, tmp_keypoint, tmp_descriptor);
			allDescriptors.push_back(tmp_descriptor);

		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}

	/*
	*	对 Feature 聚类构建单词本
	*	BuildVocabulary
	*	wordCount是指生成多少个单词，然后放在vocabulary里面
	*/
	BOWKMeansTrainer bowTrainer(wordCount);
	Mat vocabulary = bowTrainer.cluster(allDescriptors);
	printf("%d %d\n", vocabulary.cols, vocabulary.rows);



	/*
	*	对原训练集合中的每张图进行特征转换，转换为词频向量。
	*	维数为wordCount维
	*	用map分类
	*	此段功能：准备训练数据在train_features中
	*
	*
	*/
	//定义好bow特征提取和词汇本
	bowextract.setVocabulary(vocabulary);
	//将转换得到的特征向量加上labels放在all.txt里面
	ofstream tmpout("all.txt"); 
	//对每幅图生成一个向量
	for (int i = 0; i < 19; i++){
		long hFile = 0;
		string pathName, exdName;
		struct _finddata_t fileInfo;
		//String tmpname = to_string(i);
		String TmpPath = RootPath + ClassPath[i];
		if ((hFile = _findfirst(pathName.assign(TmpPath).append("\\*").c_str(), &fileInfo)) == -1) {
			return 0;
		}
		do {
			if (fileInfo.name[0] != ClassPath[i][1]) continue;
			Mat tmp_image = imread(TmpPath + "/" + fileInfo.name);
			vector<KeyPoint> tmp_keypoint;
			siftdtc.detect(tmp_image, tmp_keypoint);
			//surfdtc.detect(tmp_image, tmp_keypoint);
			Mat tmp_bowdescriptor;
			bowextract.compute(tmp_image, tmp_keypoint, tmp_bowdescriptor);//tmp_descriptor就是每张图的bow直方图表示
			//归一化  
			normalize(tmp_bowdescriptor, tmp_bowdescriptor, 1.0, 0.0, NORM_MINMAX);
			//写入文件
			for (int r = 0; r < tmp_bowdescriptor.rows; r++){
				for (int c = 0; c < tmp_bowdescriptor.cols; c++){
					int data = tmp_bowdescriptor.at<int>(r, c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
					tmpout << i << "," << data << ",";
				}
				tmpout << endl;
			}
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}
	return 0;
}
