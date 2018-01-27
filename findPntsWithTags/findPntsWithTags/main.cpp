#include"SharedHead.h"
#include"findPntsWithTags.h"

//int main01()
//{
//	int radius = 7.5;
//	int width = 16;
//	Mat circle = CoreAlgorithm::circleImg(radius, width);
//	Mat image = Mat(1500, 1500, CV_8UC1, Scalar(255));
//	Point2i offset = Point2i(0, 0);//��ͼ��ռ���ƽ������������ƽ����
//	vector<Point2i> PointVec;
//	PointVec.push_back(Point2i(100, 150));//1��
//	PointVec.push_back(Point2i(600, 100));//2��
//	PointVec.push_back(Point2i(1100, 150));//3��
//	PointVec.push_back(Point2i(600, 150));//4��
//	PointVec.push_back(Point2i(50, 150));//5��
//	PointVec.push_back(Point2i(150, 150));//6��
//	PointVec.push_back(Point2i(100, 100));//7��
//	PointVec.push_back(Point2i(100, 200));//8��
//	PointVec.push_back(Point2i(550, 100));//9��
//	PointVec.push_back(Point2i(650, 100));//10��
//	PointVec.push_back(Point2i(1100, 100));//11��
//	PointVec.push_back(Point2i(1052, 134));//12��
//	PointVec.push_back(Point2i(1147, 137));//13��
//	PointVec.push_back(Point2i(1070, 190));//14��
//	PointVec.push_back(Point2i(1129, 190));//15��
//	for (size_t i = 0; i < PointVec.size(); i++)
//	{
//		if (!CoreAlgorithm::copyCircleToImg(circle, offset, PointVec[i], image))
//			return 1;
//	}
//	cv::namedWindow("result", 1);
//	cv::imshow("result", image);
//	cv::waitKey();
//	imwrite("D:/1.bmp",image);
//	cv::Rect mask = cv::Rect(Point2f(0, 0),image.size());
//	vector<RotatedRect> findResults;
//	double precisionlevel = 5;
//	bool multi = 0;
//	int kenelsize = 5;
//	bool Result1 = CoreAlgorithm::findEllipses(image, mask, findResults, precisionlevel, multi, kenelsize);
//	vector<Point2f>  centerPnt;
//	vector<float> longaxisRadius;
//	for (int i = 0; i <findResults.size(); i++)
//	{
//		centerPnt.push_back(Point2f(findResults[i].center.x, findResults[i].center.y));
//		float length = 1.0 / 2 * (findResults[i].size.width + findResults[i].size.height);
//		longaxisRadius.push_back(length);
//	}
//	vector<TagPoint2f>  TagPnts;
//	float firstFeaturelength;
//	double gamma=double(50.0/7);
//	bool  Result2=CoreAlgorithm::findPntsWithTags(centerPnt,longaxisRadius,TagPnts, firstFeaturelength,gamma);
//	return 0;
//
//}




////////////�˺�������ʶ���������еı�־�㣨15����־�㣩
int main()
{
	Mat image = imread("E:/photos/twoCameras/Left/1.bmp");
	/*namedWindow("��־��ͼƬ", 2);
	imshow("��־��ͼƬ", image);
	waitKey();*/
	cv::Rect mask = cv::Rect(Point2f(0, 0), image.size());
	vector<RotatedRect> findResults;
	double precisionlevel = 5;
	bool multi = 0;
	int kenelsize = 5;
	bool Result1 = CoreAlgorithm::findEllipses(image, mask, findResults, precisionlevel, multi, kenelsize);
	vector<Point2f>  centerPnt;
	vector<float> longaxisRadius;
	
 	
	for (int i = 0; i < findResults.size(); i++)
	{
		if (findResults[i].size.width >8 && findResults[i].size.height > 8)
		{
			/////����������Բ���Ƴ���
		
			findResults[i].size.width *= 2;////�ɴ˿��Կ�����findEllipse������������Բֻ�ǰ��᳤
			findResults[i].size.height *= 2;
			cv::ellipse(image, findResults[i], Scalar(0, 255, 0), 1);
			cv::namedWindow("rightResult", WINDOW_NORMAL);
			cv::imshow("rightResult", image);
			cv::waitKey(6000);
			////////////////

			centerPnt.push_back(Point2f(findResults[i].center.x, findResults[i].center.y));
			float length = 1.0 / 2 * (findResults[i].size.width + findResults[i].size.height);
			longaxisRadius.push_back(length);
		}
		else
		{
			continue;
		}
	
	}
	vector<TagPoint2f>  TagPnts;
	float firstFeaturelength;
	double gamma = double(50.0 / 7);
	bool  Result2 = CoreAlgorithm::findPntsWithTags(centerPnt, longaxisRadius, TagPnts, firstFeaturelength, gamma);
	int num;
	for (size_t i = 0; i < TagPnts.size(); i++)
	{
		num =i+1;
		/////////�������ı�־����Ƴ���
		CoreAlgorithm::DrawEllipse(image, TagPnts[i][1], TagPnts[i][2], longaxisRadius[i], longaxisRadius[i],0,1);
		string Numbers = to_string(num);
		cv::putText(image, Numbers, Point(TagPnts[i][1], TagPnts[i][2]), CV_FONT_HERSHEY_COMPLEX, 1,cv::Scalar(255,0,0));
		cv::namedWindow("��־��", 2);
		cv::imshow("��־��", image);
		cv::waitKey(6000);
	}
	return 0;
}


