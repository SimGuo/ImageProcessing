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
double PresetLabels[5][19] = {
	{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 },
	{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0 },
	{ 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0},
	{ 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0},
	{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0} };
string ClassPath[19] = { "/car", "/city", "/dog", "/earth", "/fireworks", "/flowers", "/fruits",
"/glass", "/gold", "/gun", "/goldenfish", "/shoe", "/teapot", "/bear", "/dragonfly", "/football", "/plane", "/sky", "/worldcup" };
int wordCount = 30; //生成多少个单词
CvSVM mySVM[5];
/*
*	SIFT、SURF特征探测器
*	SIFT特征和SURF提取成向量的结构
*	allDescriptors中是所有的图片对应的sift特征的并集
*/
SiftFeatureDetector  siftdtc;
SurfFeatureDetector surfdtc;
SiftDescriptorExtractor siftextract;
SurfDescriptorExtractor surfextract;


Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
BOWImgDescriptorExtractor bowextract(extractor, matcher);


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
			int data = matData.at<int>(r, c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
			outFile << data << ",";   //每列数据用,隔开
		}
		outFile << endl;  //换行  
	}

	return (retVal);
}
int Myclasify(String filename){
	cout << filename << " responce: ";
	Mat tmp_image = imread(filename), tmp_bowdescriptor;
	vector<KeyPoint> tmp_keypoint;
	//siftdtc.detect(tmp_image, tmp_keypoint);
	surfdtc.detect(tmp_image, tmp_keypoint);
	bowextract.compute(tmp_image, tmp_keypoint, tmp_bowdescriptor);
	normalize(tmp_bowdescriptor, tmp_bowdescriptor, 1.0, 0.0, NORM_MINMAX);
	for (int i = 3; i < 5; i++){
		float response = mySVM[i].predict(tmp_bowdescriptor);
		cout << response << " ";
	}
	cout << endl;
	return 0;
}


int main(int argc, char* argv[]){	
	
	Mat allDescriptors; 

	/*	
	 *	提取图片的SIFT特征到txt文件的完整流程
	 *	此处SIFT未经过筛选
	 *
	 */
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
			Mat tmp_image = imread(TmpPath +"/" + fileInfo.name);
			printf("%s\n", fileInfo.name);
			vector<KeyPoint> tmp_keypoint;
			//用探测器探测到SIFT，但keypoint只是点的位置，还没有处理成向量
			//siftdtc.detect(tmp_image, tmp_keypoint);					
			surfdtc.detect(tmp_image, tmp_keypoint);
			//将对应的特征内容写入mat
			Mat tmp_descriptor;												
			//siftextract.compute(tmp_image, tmp_keypoint, tmp_descriptor);
			surfextract.compute(tmp_image, tmp_keypoint, tmp_descriptor);
			allDescriptors.push_back(tmp_descriptor);

			/*
			//将SIFT特征写入文件
			char filename[30];
			_splitpath(fileInfo.name, NULL, NULL, filename, NULL);
			String outPath;
			outPath.assign(filename);
			cout << outPath << " ";
			outPath = TmpPath + "/sift/" + outPath + ".csv";
			cout << WriteData(outPath, tmp_descriptor) << endl; //将mat内容写入文件
			*/

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
	map<int, Mat> train_features;
	bowextract.setVocabulary(vocabulary);
	//对每幅图生成一个向量
	for (int i = 0; i < 19; i++){
		long hFile = 0;
		string pathName, exdName;
		struct _finddata_t fileInfo;
		String tmpname = to_string(i);
		ofstream tmpout(tmpname + ".txt");
		String TmpPath = RootPath + ClassPath[i];
		if ((hFile = _findfirst(pathName.assign(TmpPath).append("\\*").c_str(), &fileInfo)) == -1) {
			return 0;
		}
		do {
			if (fileInfo.name[0] != ClassPath[i][1]) continue;
			Mat tmp_image = imread(TmpPath + "/" + fileInfo.name);
			vector<KeyPoint> tmp_keypoint;
			//siftdtc.detect(tmp_image, tmp_keypoint);
			surfdtc.detect(tmp_image, tmp_keypoint);
			Mat tmp_bowdescriptor;
			bowextract.compute(tmp_image, tmp_keypoint, tmp_bowdescriptor);//tmp_descriptor就是每张图的bow直方图表示
			normalize(tmp_bowdescriptor, tmp_bowdescriptor, 1.0, 0.0, NORM_MINMAX);
			for (int r = 0; r < tmp_bowdescriptor.rows; r++){
				for (int c = 0; c < tmp_bowdescriptor.cols; c++){
					int data = tmp_bowdescriptor.at<int>(r, c);
					tmpout << data << ",";
				}
				tmpout << endl;  //换行  
			}
			//归一化  
			normalize(tmp_bowdescriptor, tmp_bowdescriptor, 1.0, 0.0, NORM_MINMAX);
			//写入文件
			for (int r = 0; r < tmp_bowdescriptor.rows; r++){
				for (int c = 0; c < tmp_bowdescriptor.cols; c++){
					int data = tmp_bowdescriptor.at<int>(r, c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
					tmpout << data << ",";   
				}
				tmpout << endl;
			}
			train_features[i].push_back(tmp_bowdescriptor); //train_features准备训练数据
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
	}
	
		
		
	/*
	 *	训练分类器
	 *	设置特征和对应标签
	 *	设置支持向量机的参数（Set up SVM's parameters）
	 */
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	//训练五个svm分类器
	for (int i = 0; i < 5; i++){
		Mat trainingDataMat(0, train_features.at(0).cols, train_features.at(0).type());
		Mat responses(0, 1, CV_32SC1);

		for (auto itr = train_features.begin(); itr != train_features.end(); ++itr){
			trainingDataMat.push_back(itr->second);
			if (PresetLabels[i][itr->first] == 1.0){
				Mat response(itr->second.rows, 1, CV_32SC1, Scalar::all(1.0));
				responses.push_back(response);
			}
			else{
				Mat response(itr->second.rows, 1, CV_32SC1, Scalar::all(-1.0));
				responses.push_back(response);
			}
		}
		//if (i < 3) continue;
		mySVM[i].train(trainingDataMat, responses, Mat(), Mat(), params);
	}


	/*
	 *	test部分
	 *	用SVM对测试集进行分类
	 */
	//转化得到一个向量
	String testImgPath;
	String testRootPath = "C://Users/Lenovo/Desktop/images/testing"; //每种挑了8张的测试集
	for (int i = 0; i < 3; i++){
		int cntRight = 0;
		
		testImgPath = testRootPath + ClassPath[i] + "_0000.jpg"; 
		Myclasify(testImgPath);

		testImgPath = testRootPath + ClassPath[i] + "_0001.jpg";
		Myclasify(testImgPath);
		
		testImgPath = testRootPath + ClassPath[i] + "_0002.jpg";
		Myclasify(testImgPath);
		
		testImgPath = testRootPath + ClassPath[i] + "_0003.jpg";
		Myclasify(testImgPath);
		
		testImgPath = testRootPath + ClassPath[i] + "_0004.jpg";
		Myclasify(testImgPath);
		
		testImgPath = testRootPath + ClassPath[i] + "_0005.jpg";
		Myclasify(testImgPath);
		
		testImgPath = testRootPath + ClassPath[i] + "_0006.jpg";
		Myclasify(testImgPath);
		
		testImgPath = testRootPath + ClassPath[i] + "_0007.jpg";
		Myclasify(testImgPath);
	}

	/*	OpenCV提供了 两种Matching方式 ：
		Brute-force matcher (cv::BruteForceMatcher)
		Flann-based matcher (cv::FlannBasedMatcher)
		Brute-force matcher就是用暴力方法找到点集一中每个descriptor在点集二中距离最近的 descriptor；
		Flann-based matcher 使用快速近似最近邻搜索算法寻找。
		为了提高检测速度，你可以调用matching函数前，先训练一个matcher。
		训练过程可以首先使用cv:: FlannBasedMatcher来优化，为 descriptor建立索引树，
		这种操作将在匹配大量数据时发挥巨大作用（比如在上百幅图像的数据集中查找匹配图像）。
		而 Brute-force matcher在这个过程并不进行操作，它只是将train descriptors保存在内存中。
	*/
	/*
	FlannBasedMatcher matcher;
	//BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	Mat img_matches;
	
	matcher.match(descriptor1, descriptor2, matches);
	drawMatches(img1, kp1, img2, kp2, matches, img_matches);
	imshow("matches", img_matches);
	*/
	return 0;
	
}
