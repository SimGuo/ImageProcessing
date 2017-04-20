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
int wordCount = 4000; 
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



int extract_features(){

	//--------读取所有图片.得到bowTrainer：所有surf特征的合集
	cout << "extracting surf feature from the pictures....." << endl;
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
	ofstream tmpout("all.csv");
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
			cout << fileInfo.name << endl;
			vector<KeyPoint> tmp_keypoint;
			siftdetector.detect(tmp_image, tmp_keypoint);
			//surfdetector.detect(tmp_image, tmp_keypoint);
			Mat tmp_bowdescriptor;
			bowDesExtrac.compute(tmp_image, tmp_keypoint, tmp_bowdescriptor);//tmp_descriptor就是每张图的bow直方图表示
			for (int r = 0; r < tmp_bowdescriptor.rows; r++){
				tmpout << i;
				for (int c = 0; c < tmp_bowdescriptor.cols; c++){
					float data = tmp_bowdescriptor.at<float>(r, c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
					tmpout << "," << data;
				}
				tmpout << endl;
			}
			if (tmp_bowdescriptor.empty()) continue;
			//trainingData.push_back(tmp_bowdescriptor);
			//labels.push_back((float)i);
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}

}
int WriteData(string fileName, cv::Mat& matData) {
	int retVal = 0;

	// 检查矩阵是否为空  
	if (matData.empty()){
		cout << "矩阵为空" << endl;
		retVal = 1;
		return (retVal);
	}

	// 打开文件  
	ofstream outFile(fileName.c_str(), ios_base::out);  //按新建或覆盖方式写入  
	if (!outFile.is_open()){
		cout << "打开文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 写入数据  
	for (int r = 0; r < matData.rows; r++){
		for (int c = 0; c < matData.cols; c++){
			float data = matData.at<float>(r, c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
			outFile << data << ",";   //每列数据用,隔开
		}
		outFile << endl;  //换行  
	}

	return (retVal);
}


int main(){

	//--------第一波运行main得到all,csv.分开是为了调试svm的参数
	//--------提取特征写到文件all.csv中（在测试的时候直接读根据all.csv里面划分的测试集和训练集就可以了。
	//extract_features();


	//--------第二波运行main相应的集合.
	//--------将all.csv里面的不同的类的数据分开成不同的测试集和训练集
	//split_train_test();

	CvMLData mlData;						
	mlData.read_csv("all.csv");
	mlData.set_response_idx(0);				//表示csv文件中第一行是标签
	const CvMat* alldata = mlData.get_values();
	Mat myallData(alldata);
	Mat labelData = mlData.get_responses();	//得到所有的数据和相应的标签
	/*
	Mat ClassData[19];
	for (int i = 0; i < 19; i++){
		Mat Standard(0, wordCount, CV_32F);
		ClassData[i] = Standard;
	}
	for (int i = 0; i < myallData.rows; i++){//ClassData里面是每个类的所有数据
		Mat samplemat(1, wordCount, CV_32F);
		for (int j = 0; j < wordCount; j++){
			samplemat.at<float>(j) = myallData.at<float>(i, j + 1);
		}
		float label = labelData.at<float>(i);
		int type = (int)label;
		ClassData[type].push_back(samplemat);
	}
	for (int i = 0; i < 19; i++){
		WriteData(to_string(i) + ".csv", ClassData[i]);
	}
	//--------自己把all.csv每一类划分为十份，然后读取相应的数据
	*/

	//对所有的数据进行划分
	CvTrainTestSplit spl((float)0.9);				//90%作为训练集
	mlData.set_train_test_split(&spl);		//随机分出来percent的内容作为训练样本
	const CvMat* traindata_idx = mlData.get_train_sample_idx();
	const CvMat* testdata_idx = mlData.get_test_sample_idx();
	Mat mytrainDataidx(traindata_idx);
	Mat mytestDataidx(testdata_idx);
	//获取训练集和测试集
	Mat mytrainData(mytrainDataidx.cols, wordCount, CV_32F);
	Mat mytrainLabel(mytrainDataidx.cols, 1, CV_32S);
	Mat mytestData(mytestDataidx.cols, wordCount, CV_32F);
	Mat mytestLabel(mytestDataidx.cols, 1, CV_32S);
	for (int i = 0; i < mytrainDataidx.cols; i++){	//写训练集
		mytrainLabel.at<int>(i) = labelData.at<float>(mytrainDataidx.at<int>(i));
		for (int j = 0; j < wordCount; j++){
			mytrainData.at<float>(i, j) = myallData.at<float>(mytrainDataidx.at<int>(i), j + 1);
		}
	}
	for (int i = 0; i < mytestDataidx.cols; i++){
		mytestLabel.at<int>(i) = labelData.at<float>(mytestDataidx.at<int>(i));
		for (int j = 0; j < wordCount; j++){
			mytestData.at<float>(i, j) = myallData.at<float>(mytestDataidx.at<int>(i), j + 1);
		}
	}




	//--------SVM训练
	//设置参数
	CvSVMParams params;
	params.kernel_type = CvSVM::LINEAR;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1e5, 1e-7);
	CvSVM svm;

	cout << "training the SVM" << endl;
	bool res = svm.train_auto(mytrainData, mytrainLabel, Mat(), Mat(), params);
	cout << "Processing evaluation data..." << endl;

	//--------SVM测试
	int rightcnt = 0;
	for (int i = 0; i < mytestData.rows; i++)
	{
		Mat samplemat(mytestData, Range(i, i + 1));
		float response = svm.predict(samplemat, true);
		float truelabel = mytestLabel.at<int>(i);
		if (response == truelabel){
			rightcnt++;
		}
	}
	double accurRate = 100.0 * (float)rightcnt / mytestData.rows;
	cout << "accuracy by the opencv svm is ..." << accurRate << "%" << endl;
	return 0;
}
