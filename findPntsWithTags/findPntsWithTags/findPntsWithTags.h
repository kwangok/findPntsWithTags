#include"SharedHead.h"
#include "sstream"
class CoreAlgorithm
{
public:
	//ellipse fitting
	static bool findEllipses(const Mat img, const cv::Rect mask, vector<RotatedRect>& findResults, const double precisionlevel, bool multi, int kenelsize);
	static bool DualConicFitting(vector<Point2d>areaGrad, vector<Point2d>areaPos, Mat& dC, Mat& precision, double& angleIncertitude);
	static bool conicaEllipseTostd(const Mat ellipsePara, RotatedRect& result);
	static bool MultiEllipseFitting(vector<vector<cv::Point2d>>areaGrads,
		vector<vector<cv::Point2d>>areaPoses,
		vector<Mat>& dCs, vector<Mat>& precisions,
		vector<double> angleIncertitudes);
	static bool findPntsWithTags(vector<Point2f>& centerPnt, vector<float>& longaxisRadius, vector<TagPoint2f>& TagPnts, float &firstFeaturelength, const double gamma);
	static double  distancePoints(Point2f Pnt_1, Point2f Pnt_2);
	static Mat circleImg(int radius, int width);
	static bool copyCircleToImg(const Mat circleImg, Point2i offset, Point2i location, Mat &writeImg);
	static Mat DrawEllipse(Mat img, double EllipseCenter_x, double EllipseCenter_y, double EllipseLong_axis, double EllipseShort_axis, double angle, int ksizeWidth);
	
};