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

const string RootPath = "F://Current/MachineLearning/imagedata";

string ClassPath[19] = { "/car", "/city", "/dog", "/earth", "/fireworks", "/flowers", "/fruits",
"/glass", "/gold", "/gun", "/goldenfish", "/shoe", "/teapot", "/bear", "/dragonfly", "/football", "/plane", "/sky", "/worldcup" };

void dir(string path)
{
	long hFile = 0;
	struct _finddata_t fileInfo;
	string pathName, exdName;

	if ((hFile = _findfirst(pathName.assign(path).append("\\*").c_str(), &fileInfo)) == -1) {
		return;
	}
	do {
		cout << fileInfo.name << endl;
	} while (_findnext(hFile, &fileInfo) == 0);
	_findclose(hFile);
	return;
}


int WriteData(string fileName, cv::Mat& matData) {
	
	int retVal = 0;

	// 检查矩阵是否为空  
	if (matData.empty())
	{
		cout << "矩阵为空" << endl;
		retVal = 1;
		return (retVal);
	}

	// 打开文件  
	ofstream outFile(fileName.c_str(), ios_base::out);  //按新建或覆盖方式写入  
	if (!outFile.is_open())
	{
		cout << "打开文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 写入数据  
	for (int r = 0; r < matData.rows; r++)
	{
		for (int c = 0; c < matData.cols; c++)
		{
			int data = matData.at<int>(r, c);    //读取数据，at<type> - type 是矩阵元素的具体数据格式  
			outFile << data << ",";   //每列数据用,隔开
		}
		outFile << endl;  //换行  
	}

	return (retVal);
}


int main(int argc, char* argv[]){


	/*
	 *	SIFT特征探测器 
	 *	SIFT特征提取成向量的结构
	 */
	SiftFeatureDetector  siftdtc;
	SiftDescriptorExtractor extractor;							

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
			Mat tmp_image = imread(TmpPath +"/" + fileInfo.name);		//读入图片
			vector<KeyPoint> tmp_keypoint;
			siftdtc.detect(tmp_image, tmp_keypoint);							//用探测器探测到SIFT，但keypoint只是点的位置，还没有处理成向量
			Mat tmp_descriptor;												//将对应的特征内容写入mat
			extractor.compute(tmp_image, tmp_keypoint, tmp_descriptor);

			char filename[30];
			_splitpath(fileInfo.name, NULL, NULL, filename, NULL);
			String outPath;
			outPath.assign(filename);
			cout << outPath << " ";
			outPath = TmpPath + "/sift/" + outPath + ".csv";
			cout << WriteData(outPath, tmp_descriptor) << endl; //将mat内容写入文件

		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);		
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
