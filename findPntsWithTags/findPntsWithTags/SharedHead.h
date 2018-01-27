#pragma once
//////////////////////����////////////////////////////
////////////////////////////////////////////////////
#include "opencv.hpp"
#include <iostream>
using namespace cv;
#include "fstream"
#include "time.h"
#include"stdlib.h"
#include "algorithm"
#include "math.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cstddef>



////////////////////////////////////////////////////
///////////////////////////////////////////////////
#include <string>
#include <vector>
#include "cv.h"
#include <iomanip>

using namespace std;

using cv::Mat;
using cv::Vec4f;
using cv::Point3f;
using cv::Vec3f;
using cv::Point2f;
using cv::Size;
using cv::Point;
using cv::Scalar;
using cv::RotatedRect;
//28.152,12,7
//26.928 11 8
//37.2 11 5
#define SQUARE_SIZE 30			//<���̸񷽸�ߴ�
#define WIDTH_CORNER_COUNT 14		//<���߽ǵ����
#define HEIGHT_CORNER_COUNT 11		//<�̱߽ǵ����
#define CAMERA_NUM 2				//<�������������
////chengwei added
#define HEADRADIUS 0.99999f;
//��ʲ�ͷ�뾶�������뾶����

typedef Vec3f TagPoint2f;
typedef Vec4f TagPoint3f;
typedef Point3f SurfaceNormal;

//#define BLUR
//#define CANNY
//#define THRESHOL

//opencv3.0
#define TRUE 1
#define FALSE 0

//enum PointTag
//{
//    TAG1,
//    TAG2,
//    TAG3,
//    TAG4,
//    TAG5,
//    TAG6,
//    TAG7,
//	TAG8
//	
//
//};
enum PointTag
{
	TAG1=1,
	TAG2,
	TAG3,
	TAG4,
	TAG5,
	TAG6,
	TAG7,
	TAG8,
	TAG9,
	TAG10,
	TAG11,
	TAG12,
	TAG13,
	TAG14,
	TAG15
};

//enum PointTagv
//{
//    TAG1,
//    TAG2,
//    TAG3,
//    TAG4,
//    TAG5,
//    TAG6,
//    TAG7,
//	TAG8
//};

enum SystemMode
{
    SINGLE_CAM,
    DOUBLE_CAM,
    TRIPLE_CAM
};

enum CamPosition
{
    LEFT,
    MID,
    RIGHT
};

struct RT
{
    cv::Mat R; //<3X1
    cv::Mat T; //<3X1
};


class CalibrationData
{
public:
    CalibrationData() {}

    void ErasePoint(unsigned int frameNum, unsigned int pointNum)//ɾ��frameNum�еĵ�pointNum���㣨0Ϊ��һ���㣩
    {
        vector< vector<Point3f> >::iterator IterPoint3fs;
        vector< vector<Point2f> >::iterator IterPoint2fs;
        vector<Point3f>::iterator IterPoint3f;
        vector<Point2f>::iterator IterPoint2f;
       unsigned int i = 0;
        for(vector<int>::iterator IterFrame = frameNumList.begin(); IterFrame != frameNumList.end(); i++)
        {
            if (*IterFrame == frameNum)
            {
                IterPoint3fs = plane3dPntsVec.begin() + i;
                IterPoint2fs = plane2dPntsVec.begin() + i;
                if (i >= plane3dPntsVec.size() || i >= plane2dPntsVec.size())
                    return;
                IterPoint3f = IterPoint3fs->begin();
                IterPoint2f = IterPoint2fs->begin();
                if (pointNum >= plane3dPntsVec[i].size() || pointNum >= plane2dPntsVec[i].size())
                    return;
                IterPoint3fs->erase(IterPoint3f + pointNum);
                IterPoint2fs->erase(IterPoint2f + pointNum);
                break;
            }
            else
                IterFrame++;
        }
    }


    int imgHeight;								//<ͼ��ĸ�
    int imgWidth;								//<ͼ��Ŀ�
    vector<int> frameNumList;					//<ͼ�������
	int cols;									//<ͼ��е������
	int rows;									//<ͼ��е������
	double radius;	                             //Բ�뾶
    //std::vector<int> CornerNumPerFrame;       //<ÿһ֡ͼ��ǵ����
    vector< vector<Point3f> > plane3dPntsVec;		//<�궨����άƽ������
    vector< vector<Point2f> > plane2dPntsVec;		//<�궨���ά����

};


class CamPara
{
public:
    CamPara()
    {
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                CameraIntrinsic[i][j]=0;
        CameraIntrinsic[2][2] = 1;
        for(int i=0;i<4;i++) {DistortionCoeffs[i]=0;kcError[i]=0;}
        ReprojectionError[0]=0;ReprojectionError[1]=0;
        fcError[0]=0;fcError[1]=0;
        ccError[0]=0;ccError[1]=0;
    }
    double CameraIntrinsic[3][3];	//<����ڲ���
    double DistortionCoeffs[4];		//<����������
    std::vector<RT> imgRTVec;		//<�궨�������
    Mat parallelCamR;					//<ƽ�������ͼ��R
    double fcError[2];
    double ccError[2];
    double kcError[4];
    double ReprojectionError[2];		//<��ͶӰ����׼��
    std::vector<double> reprojectNormErr;
    double totalReproNormErr;
};

///////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// ��ΰ ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

enum
{
    PNP_ITERATIVE = 0,
    PNP_EPNP = 1, // F.Moreno-Noguer, V.Lepetit and P.Fua "EPnP: Efficient Perspective-n-Point Camera Pose Estimation"
    PNP_P3P = 2, // X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang; "Complete Solution Classification for the Perspective-Three-Point Problem"
    //����������Ҫopencv3.0���ϰ汾��֧��
    PNP_DLS = 3, //Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis. ��A Direct Least-Squares (DLS) Method for PnP��.
    PNP_UPNP = 4 //Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto, F.Moreno-Noguer.
    //��Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation��.
    //In this case the function also estimates the parameters f_x and f_y assuming that both have the same value.
    //Then the cameraMatrix is updated with the estimated focal length.
};

enum CAMBEHAVIOR
{
    CAMCALIBRATON,
    LIGHTPENCALIBRATION,
    MESSURE,
	COMCALIBRATION

};

enum MESMODEL
{
    MESSINGLEPOINT,
    MESPLANEFIRST,
    MESPLANESECOND,
    MESCIRCLECYLINDER,
    MESPOINTTOPOINT,
	COMMESSURE
};

struct ImageSaveDir
{
    string calibrateCamDir;
    string calibrateProbeDir;
    string probePntDir;
};

class Mat34
{
public:
    Mat34()
    {
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<4;j++)
            {
                mat[i][j] = 0;
            }
        }
    }
    double mat[3][4];
};

class CamGroupPara
{
public:
    CamGroupPara()
    {
        for(int i=0;i<3;i++)
        {
            left2MidRotVector[i] = 0;
            left2MidTraVector[i] = 0;
            right2MidRotVector[i] = 0;
            right2MidTraVector[i] = 0;
            right2LeftRotVector[i] = 0;
            right2LeftTraVector[i] = 0;
        }
    }

public:
    double left2MidRotVector[3];
    double left2MidTraVector[3];
    double right2MidRotVector[3];
    double right2MidTraVector[3];
    double right2LeftRotVector[3];
    double right2LeftTraVector[3];
};

//����ÿ�β����Ĳ�ͷ�������꣨��������ϵ�µģ��Ͳ�������ķ���
struct CenterFeartures
{
    Point3f center1;
    Point3f center2;
    Point3f center3;
    SurfaceNormal feature;//��������ķ���
};

//�����뾶�����ı����������������꣨�ڲ�������ϵ�µģ��Լ���������ķ���
struct MessureResult
{
    Point3f point1;
    Point3f point2;
    Point3f point3;
    SurfaceNormal feature;//��������ķ���
};

//��ʱ궨����
struct LightPenPara
{
    vector<TagPoint3f> FeatursPoints;
    Point3f CenterPoint;
};

//ƽ��ģ��
class Plane
{
public:
    SurfaceNormal normal;
    Point3f orignal;
};

//����ģ��
class CircleCylinder
{
public:
    SurfaceNormal axisNormal;//��λ����
    Point3f orignal;//λ�������ϵ�һ��
    float r;//Բ���뾶
};

class Line
{
public:
    SurfaceNormal normal;//��������
    Point3f orignal;//ֱ����һ��

};

///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// ֣���� //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

#define MultiNum  5		//��СԲ����Ŵ���
#define ALLOWERROR 0.09		//�����������
#define ThreshLevel 60	//Բ��ȡ��ֵ
#define AreaMax 10000		//����ɸѡ������ֵ
#define AreaMin 20			//����ɸѡ�����Сֵ
#define RATE    4			//��СԲ�������
#define LRATE    4         //��Բ������ȣ����б�б궨ʱLRATEӦѡ��ֵ�Ƽ�4������������ʶ��ʱѡСֵ���Ƽ�2~2.2

//��б궨���
#define FIR_TO_TW 101
#define FIR_TO_TH 102
#define FIR_TO_FO 103
#define FIR_TO_FI 104

enum FeaCircleFlag   //����Բ���
{
    CIR_0=0,   //���ʾ��Ϊ����ʶ
    CIR_1=1,
    CIR_2,
    CIR_3,
    CIR_4,
    CIR_5,
    CIR_6,
    CIR_7,
    CIR_a,
    CIR_b,
    CIR_c,
};
//enum AreaTag
//{
//	AREA_1=1,
//	AREA_2,
//	AREA_3,
//};
enum FaceFlag  //���ű�ʶ
{
    Face_1=1,   //���ʾ��δ����ʶ
    Face_2,
    Face_3,
    Face_4,
    Face_5,
    Face_6,
};

struct FeaCircle  //�������Բ��Ϣ�Ľṹ��
{
    Point2f center;
    float height,width;
    enum FeaCircleFlag flag;
};

struct SH_Ellipse//�����ȡ��Բ��Ϣ�ṹ��
{
    Point2f center;
    float macroaxis,brachyaxis;
};

struct circle_inf		//����Բ������Ϣ�ṹ��
{
    Point2f center;
    float height;
    float width;
    vector<float> dis2othercir;			//������Բ�ľ���
    enum FeaCircleFlag flag;			//Բ�����ǣ�
    int markflag;						//����ʶ��һ��Բ�ı��
    vector<int> cirflag;				//���ڴ���ڸ�Բ����С��1.25D��Բ�ı��
    double dist2large1[2];				//��һ�Ŵ�Բ�ľ���,һ���洢X����һ������Y����
    int dist2large1Flag;				//�����洢����Բ��һ�Ŵ�Բ����Ĵ�С˳���־�������ҳ�����һ�Ŵ�Բ�����������Բ
};

struct mark_area				 //�洢�����Ľṹ��
{
    //���صȺŲ���
    mark_area& operator=(mark_area& value)
    {
        large_circle = value.large_circle;
        small_circle = value.small_circle;
        Face=value.Face;
        return *this;
    }
    vector <Point2f> large_circle;
    vector <Point2f> small_circle;
    enum FaceFlag Face;
};

struct StereoEllipse
{
    CvPoint3D32f  NormalVector;
    CvPoint3D32f  center;
    double r;
};

class ObjectPara
{
public:
    ObjectPara()
    {
        vector<Point3f> FacePnt;
        for (int i=0;i<7;i++)
        {
            FacePnt.push_back(Point3f(0,0,0));
        }
        for (int i=0;i<5;i++)
        {
            FacePnts[i] = FacePnt;
        }
        for (int i=0;i<4;i++)
        {
            RT2one[i]=cv::Mat::zeros(4,4,CV_64FC1);
        }
    }
    ////���صȺŲ���
    //ObjectPara& operator=(ObjectPara& value)
    //{
    //	for (int i =0;i<5;i++)
    //	{
    //		for (int j =0 ;j<value.FacePnts->size(); j++)
    //		{
    //			FacePnts[i].push_back(value.FacePnts[i][j]);
    //		}
    //	}
    //	for ( int i=0;i<4;i++)
    //	{
    //		value.RT2one[i].copyTo(RT2one[i]);
    //	}
    //	return *this;
    //}
    vector<Point3f> FacePnts[5];
    Mat RT2one[4];
};

typedef vector <RotatedRect> EllipseBox;
typedef vector <mark_area> MarkArea;

