#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <fstream>
#include <io.h>
#include <vector>
#include <ml.h>
using namespace std;
using namespace cv;


//const string RootPath = "C://Users/Lenovo/Desktop/images/training";  //简化的训练集
const string SiftBoWPath = "F://Current/MachineLearning/Imageproject/Imageproject/siftbowfeature/";
const string SurfBoWPath = "F://Current/MachineLearning/Imageproject/Imageproject/surfbowfeature/";
static int wordCount = 30; //生成多少个单词

int ReadData(String filename, Mat data){
	int retVal = 0;
	ifstream inFile(filename.c_str(), ios_base::in);
	if (!inFile.is_open()){
		cout << "打开文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	//数据的行数不定，列数为wordcount
	// 写入数据  
	while (!inFile.eof()){
		Mat linedata;// (1, wordCount, CV_32SC1);
		for (int col = 1; col <= wordCount; col++){
			int tmp; char tmpchar;
			inFile >> tmp >> tmpchar;
			linedata.at<double>(1, col) = tmp;
		}
	}
	return retVal;
}



int main(int argc, char* argv[]){
	
	//读取训练数据
	CvMLData mlData;						//包含了所有类的训练数据
	int num = mlData.read_csv("all.csv");
	cout << num << endl;

	//对所有的数据进行划分
	CvTrainTestSplit spl((float)0.9);				//90%作为训练集
	mlData.set_train_test_split(&spl);		//随机分出来percent的内容作为训练样本
	const CvMat* traindata_idx = mlData.get_train_sample_idx();
	const CvMat* testdata_idx = mlData.get_test_sample_idx();
	Mat mytrainDataidx(traindata_idx);
	Mat mytestDataidx(testdata_idx);
	
	//获取训练集和测试集
	mlData.set_response_idx(0);				//表示csv文件中第一行是标签
	const CvMat* alldata = mlData.get_values();
	Mat myallData(alldata);
	Mat labelData = mlData.get_responses();	//得到所有的数据和相应的标签
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

	//训练
	CvSVM opencv_svm;
	CvSVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
	opencv_svm.train(mytrainData, mytrainLabel, Mat(), Mat(), params);

	int k = 0;
	for (int i = 0; i<mytestData.rows; i++)
	{
		Mat samplemat(mytestData, Range(i, i + 1));
		float response = opencv_svm.predict(samplemat, wordCount);
		float truelabel = mytestLabel.at<int>(i);
		if (response * truelabel < 0){
			k++;
		}
	}
	cout << "accuracy by the opencv svm is ..." << 100.0 * (float)k / mytestData.rows << endl;


	return 0;

}
