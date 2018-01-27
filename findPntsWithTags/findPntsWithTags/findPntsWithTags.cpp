#include"SharedHead.h"
#include"findPntsWithTags.h"
bool CoreAlgorithm::findEllipses(const Mat img, const cv::Rect mask, vector<RotatedRect>& findResults, const double precisionlevel, bool multi, int kenelsize)
{
	//step-1 将图像转化成灰度图
	Mat ImageGray;
	if (img.channels() == 1)
	{
		ImageGray = img;
	}
	else
	{
		cvtColor(img, ImageGray, CV_BGR2GRAY);
	}
	ImageGray = Mat(ImageGray, mask);
	//namedWindow("window",1);
	//imshow("window",ImageGray);
	//waitKey();
	//step-2 计算图像梯度信息
	//step-2-1 生成高斯滤波器模版	
	//判断滤波片模版大小是不是奇数
	if (kenelsize % 2 == 0)
		return false;
	Mat X, Y;
	cv::Range xgv = cv::Range(-abs((kenelsize - 1) / 2), abs((kenelsize - 1) / 2));
	cv::Range ygv = xgv;
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);
	cv::repeat(cv::Mat(t_x), 1, t_y.size(), X);
	cv::repeat(cv::Mat(t_y).t(), t_x.size(), 1, Y);
	Mat GaussianKenelx(kenelsize, kenelsize, CV_64F);
	Mat GaussianKenely(kenelsize, kenelsize, CV_64F);
	for (int i = 0; i < kenelsize; i++)
	{
		for (int j = 0; j < kenelsize; j++)
		{
			GaussianKenelx.at<double>(i, j) = double(-X.at<int>(i, j)*exp(pow(X.at<int>(i, j), 2) / -2)*exp(pow(Y.at<int>(i, j), 2) / -2));
			GaussianKenely.at<double>(i, j) = double(-Y.at<int>(i, j)*exp(pow(X.at<int>(i, j), 2) / -2)*exp(pow(Y.at<int>(i, j), 2) / -2));
		}
	}
	//step-2-2 二维滤波器操作
	Mat dx, dy;
	filter2D(ImageGray, dx, CV_64F, GaussianKenelx);
	filter2D(ImageGray, dy, CV_64F, GaussianKenely);
	//step-2-3 梯度范数计算
	Mat gradientnorm, gradientnormBinary;
	magnitude(dx, dy, gradientnorm);//计算梯度范数
	double minvalue, maxvalue;
	cv::minMaxLoc(gradientnorm, &minvalue, &maxvalue);
	//step-3 高置信梯度区域的选择
	//step-3-1 针对求出的梯度矩阵进行二值化
	int thresholdvalue = int(minvalue + maxvalue / 5);
	//尝试直接二值化的方法
	//imwrite("F:/test3.jpg",ImageGray);
	//imwrite("F:/test2.jpg",gradientnorm);
	gradientnorm.convertTo(gradientnorm, CV_32F);
	double value = threshold(gradientnorm, gradientnormBinary, thresholdvalue, 255, CV_THRESH_BINARY);
	gradientnormBinary.convertTo(gradientnormBinary, CV_8UC1);

	//以上部分是不是可以通过先高斯滤波处理再使用canny算子进行边缘检测？？？？？？？？？
	//step-3-2 联通区域标识
	Mat contoursMask;
	int contoursNum = connectedComponents(gradientnormBinary, contoursMask, 8);
	contoursMask.convertTo(contoursMask, CV_16UC1);
	int type = contoursMask.type();
	contoursNum--;
	//step-3-3 数据整理
	vector<vector<Point2d>> conectAreasPos;
	vector<vector<Point2d>>  conectAreasGrad;
	conectAreasGrad.resize(contoursNum);
	conectAreasPos.resize(contoursNum);
	for (int i = 0; i < contoursMask.rows; i++)
	{
		for (int j = 0; j < contoursMask.cols; j++)
		{
			int tempnum = contoursMask.at<unsigned short>(i, j) - 1;
			if (contoursMask.at<unsigned short>(i, j) != 0)
			{
				conectAreasPos[tempnum].push_back(Point2d(double(i), double(j)));
				conectAreasGrad[tempnum].push_back(Point2d(dx.at<double>(i, j), dy.at<double>(i, j)));
			}
		}
	}
	//step-4 利用对偶椭圆算子

	vector<Mat> dCVec, precisionVec;
	vector<double> angleIncertitudeVec;
	if (!multi)
	{
		for (int i = 0; i < contoursNum; i++)
		{
			Mat dC, precision;
			double AngleIncertitude;
			if (conectAreasPos[i].size()<10)
				continue;
			if (!DualConicFitting(conectAreasGrad[i], conectAreasPos[i], dC, precision, AngleIncertitude))
				continue;
			if (precision.at<double>(0, 0) == -1 || precision.at<double>(0, 0)>precisionlevel)
				continue;
			double num1 = precision.at<double>(0, 0);
			dCVec.push_back(dC);
			precisionVec.push_back(precision);
			angleIncertitudeVec.push_back(AngleIncertitude);
		}
	}
	else
	{
		if (!MultiEllipseFitting(conectAreasGrad, conectAreasPos, dCVec, precisionVec, angleIncertitudeVec))
		{
			return false;
		}
	}
	//step-5 椭圆参数计算 Ax^2+Bxy+Cy^2+Dx+Ey+F=0
	vector<Mat> EllipsesparaVec;
	for (unsigned int i = 0; i < dCVec.size(); i++)
	{
		Mat Ellipsespara = Mat(6, 1, CV_64F);
		Mat _C = dCVec[i].inv();
		_C = _C / _C.at<double>(2, 2);
		Ellipsespara.at<double>(0, 0) = _C.at<double>(1, 1);
		Ellipsespara.at<double>(1, 0) = _C.at<double>(0, 1) * 2;
		Ellipsespara.at<double>(2, 0) = _C.at<double>(0, 0);
		Ellipsespara.at<double>(3, 0) = _C.at<double>(1, 2) * 2;
		Ellipsespara.at<double>(4, 0) = _C.at<double>(2, 0) * 2;
		Ellipsespara.at<double>(5, 0) = _C.at<double>(2, 2);
		EllipsesparaVec.push_back(Ellipsespara);
	}
	//step-6 由椭圆一般方程求解椭圆标准方程参数
	vector<RotatedRect> findResultstemp;
	for (unsigned int i = 0; i < EllipsesparaVec.size(); i++)
	{
		RotatedRect temppara;
		if (!conicaEllipseTostd(EllipsesparaVec[i], temppara))
			continue;
		findResultstemp.push_back(temppara);
	}
	for (unsigned int i = 0; i < findResultstemp.size(); i++)
	{
		findResultstemp[i].center.x += mask.x;
		findResultstemp[i].center.y += mask.y;
	}
	findResults = findResultstemp;
	return true;
}
bool CoreAlgorithm::DualConicFitting(vector<Point2d>areaGrad, vector<Point2d>areaPos, Mat& dC, Mat& precision, double& angleIncertitude)
{
	precision = Mat::zeros(1, 2, CV_64F);
	Mat a, b, c, _M;
	Mat areaGradmat = Mat(areaGrad).reshape(1);
	Mat areaPosmat = Mat(areaPos).reshape(1);
	a = areaGradmat.col(0);
	b = areaGradmat.col(1);
	Mat multitemp = areaGradmat.mul(areaPosmat);
	addWeighted(multitemp.col(0), -1, multitemp.col(1), -1, 0, c);

	//为了高精度检测，对数据进行了线性归一化
	Mat M = Mat(a.rows, 2, CV_64F);
	Mat tempb = -1 * b;
	tempb.copyTo(M.col(0));
	a.copyTo(M.col(1));
	Mat B = -1 * c;
	Mat mpts, Lnorm, Lnormt, Minvert;
	if (!solve(M, B, mpts, DECOMP_SVD))
		return false;
	Mat H = Mat::eye(3, 3, CV_64F);
	H.at<double>(0, 2) = mpts.at<double>(0, 0);
	H.at<double>(1, 2) = mpts.at<double>(1, 0);
	Mat abc = Mat(a.rows, 3, CV_64F);
	a.copyTo(abc.col(0));
	b.copyTo(abc.col(1));
	c.copyTo(abc.col(2));

	Lnorm = H.t()*abc.t();
	Lnormt = Lnorm.t();
	a = Lnormt.col(0).clone();
	b = Lnormt.col(1).clone();
	c = Lnormt.col(2).clone();
	Mat AA = Mat(5, 5, CV_64F);
	Mat BB = Mat(5, 1, CV_64F);
	Mat a2 = a.mul(a);
	Mat ab = a.mul(b);
	Mat b2 = b.mul(b);
	Mat ac = a.mul(c);
	Mat bc = b.mul(c);
	Mat c2 = c.mul(c);
	//solution par least-square
	//AA*THITA=BB;
	//求AA
	Mat aaaa = a2.mul(a2); Mat aaab = a2.mul(ab); Mat aabb = a2.mul(b2); Mat aaac = a2.mul(ac); Mat aabc = a2.mul(bc);
	Mat abab = ab.mul(ab); Mat abbb = ab.mul(b2); Mat abac = ab.mul(ac); Mat abbc = ab.mul(bc);
	Mat bbbb = b2.mul(b2); Mat bbac = b2.mul(ac); Mat bbbc = b2.mul(bc);
	Mat acac = ac.mul(ac); Mat acbc = ac.mul(bc);
	Mat bcbc = bc.mul(bc);
	AA.at<double>(0, 0) = sum(aaaa).val[0]; AA.at<double>(0, 1) = sum(aaab).val[0]; AA.at<double>(0, 2) = sum(aabb).val[0]; AA.at<double>(0, 3) = sum(aaac).val[0]; AA.at<double>(0, 4) = sum(aabc).val[0];
	AA.at<double>(1, 0) = sum(aaab).val[0]; AA.at<double>(1, 1) = sum(abab).val[0]; AA.at<double>(1, 2) = sum(abbb).val[0]; AA.at<double>(1, 3) = sum(abac).val[0]; AA.at<double>(1, 4) = sum(abbc).val[0];
	AA.at<double>(2, 0) = sum(aabb).val[0]; AA.at<double>(2, 1) = sum(abbb).val[0]; AA.at<double>(2, 2) = sum(bbbb).val[0]; AA.at<double>(2, 3) = sum(bbac).val[0]; AA.at<double>(2, 4) = sum(bbbc).val[0];
	AA.at<double>(3, 0) = sum(aaac).val[0]; AA.at<double>(3, 1) = sum(abac).val[0]; AA.at<double>(3, 2) = sum(bbac).val[0]; AA.at<double>(3, 3) = sum(acac).val[0]; AA.at<double>(3, 4) = sum(acbc).val[0];
	AA.at<double>(4, 0) = sum(aabc).val[0]; AA.at<double>(4, 1) = sum(abbc).val[0]; AA.at<double>(4, 2) = sum(bbbc).val[0]; AA.at<double>(4, 3) = sum(acbc).val[0]; AA.at<double>(4, 4) = sum(bcbc).val[0];
	//求BB
	Mat _ccaa = -1 * (c2.mul(a2)); Mat _ccab = -1 * (c2.mul(ab)); Mat _ccbb = -1 * (c2.mul(b2)); Mat _ccac = -1 * (c2.mul(ac)); Mat _ccbc = -1 * (c2.mul(bc));
	BB.at<double>(0, 0) = sum(_ccaa).val[0]; BB.at<double>(1, 0) = sum(_ccab).val[0]; BB.at<double>(2, 0) = sum(_ccbb).val[0]; BB.at<double>(3, 0) = sum(_ccac).val[0]; BB.at<double>(4, 0) = sum(_ccbc).val[0];
	if (determinant(AA) < 10e-10)
	{
		//是否没有必要做下面工作，直接return false'
		dC = Mat::ones(3, 3, CV_64F);
		dC = -1 * dC;
		precision.at<double>(0, 0) = -1;
		angleIncertitude = -1;
		return false;
	}
	//解A*THITA=BB;
	Mat w, u, vt;
	Mat sol = Mat(5, 1, CV_64F);
	if (!solve(AA, BB, sol))
		return false;
	//denormalisation
	Mat dCnorm = Mat(3, 3, CV_64F);
	dCnorm.at<double>(0, 0) = sol.at<double>(0, 0);
	dCnorm.at<double>(0, 1) = sol.at<double>(1, 0) / 2;
	dCnorm.at<double>(0, 2) = sol.at<double>(3, 0) / 2;
	dCnorm.at<double>(1, 0) = sol.at<double>(1, 0) / 2;
	dCnorm.at<double>(1, 1) = sol.at<double>(2, 0);
	dCnorm.at<double>(1, 2) = sol.at<double>(4, 0) / 2;
	dCnorm.at<double>(2, 0) = sol.at<double>(3, 0) / 2;
	dCnorm.at<double>(2, 1) = sol.at<double>(4, 0) / 2;
	dCnorm.at<double>(2, 2) = 1;
	dC = H*dCnorm*H.t();
	//误差估计
	Mat ss = sol.t();
	Mat cccc = c2.mul(c2);
	double BIB = sum(cccc).val[0];
	Mat R = (ss*AA*sol - 2 * ss*BB + BIB) / (a.rows - 5);
	double RmatValue = R.at<double>(0, 0);
	Mat cvar2_constantVariance = RmatValue*AA.inv();
	double vD = cvar2_constantVariance.at<double>(3, 3);
	double vDE = cvar2_constantVariance.at<double>(3, 4);
	double vE = cvar2_constantVariance.at<double>(4, 4);
	Mat errorMatrics = Mat(2, 2, CV_64F);
	errorMatrics.at<double>(0, 0) = cvar2_constantVariance.at<double>(3, 3);
	errorMatrics.at<double>(0, 1) = cvar2_constantVariance.at<double>(3, 4);
	errorMatrics.at<double>(1, 0) = cvar2_constantVariance.at<double>(4, 3);
	errorMatrics.at<double>(1, 1) = cvar2_constantVariance.at<double>(4, 4);
	SVD::compute(errorMatrics, w, u, vt);
	Mat diagresult;
	sqrt(w, diagresult);
	diagresult = diagresult / 4;
	precision = diagresult.t();
	angleIncertitude = atan2(vt.at<double>(1, 1), vt.at<double>(0, 1));
	return true;
}
bool CoreAlgorithm::conicaEllipseTostd(const Mat ellipsePara, RotatedRect& result)
{
	double thetarad, cost, sint, sin_squared, cos_squared, cos_sin;
	thetarad = 0.5*atan2(ellipsePara.at<double>(1, 0), (ellipsePara.at<double>(0, 0) - ellipsePara.at<double>(2, 0)));
	cost = cos(thetarad);
	sint = sin(thetarad);
	sin_squared = sint*sint;
	cos_squared = cost*cost;
	cos_sin = sint*cost;
	double Ao, Au, Av, Auu, Avv;
	Ao = ellipsePara.at<double>(5, 0);
	Au = ellipsePara.at<double>(3, 0)*cost + ellipsePara.at<double>(4, 0)*sint;
	Av = -ellipsePara.at<double>(3, 0)*sint + ellipsePara.at<double>(4, 0)*cost;
	Auu = ellipsePara.at<double>(0, 0)*cos_squared + ellipsePara.at<double>(2, 0)*sin_squared + ellipsePara.at<double>(1, 0)*cos_sin;
	Avv = ellipsePara.at<double>(0, 0)*sin_squared + ellipsePara.at<double>(2, 0)*cos_squared - ellipsePara.at<double>(1, 0)*cos_sin;
	if (Auu == 0 || Avv == 0)
	{
		//problem.  this is not a valid ellipse
		//make sure param are invalid and be easy to spot
		result.center.x = -1;
		result.center.y = -1;
		result.size.height = 0;
		result.size.width = 0;
		result.angle = 0;
		return false;
	}
	double tuCentre, tvCentre, wCentre, uCentre, vCentre, Ru, Rv;
	// ROTATED = [Ao Au Av Auu Avv]
	tuCentre = -Au / (2 * Auu);
	tvCentre = -Av / (2 * Avv);
	wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre;
	uCentre = tuCentre * cost - tvCentre * sint;
	vCentre = tuCentre * sint + tvCentre * cost;

	Ru = -wCentre / Auu;
	Rv = -wCentre / Avv;
	if (Ru < 0)
	{
		Ru = -1 * sqrt(abs(Ru));
	}
	else
	{
		Ru = sqrt(abs(Ru));
	}
	if (Rv < 0)
	{
		Rv = -1 * sqrt(abs(Rv));
	}
	else
	{
		Rv = sqrt(abs(Rv));
	}
	result.center.x = (float)uCentre;
	result.center.y = (float)vCentre;
	if (Ru < Rv)
	{
		result.size.height = (float)Ru;
		result.size.width = (float)Rv;
		result.angle = (float)thetarad;
	}
	else
	{
		result.size.height = (float)Rv;
		result.size.width = (float)Ru;
		result.angle = (float)thetarad;
	}
	return true;
}

bool CoreAlgorithm::MultiEllipseFitting(vector<vector<cv::Point2d>>areaGrads,
	vector<vector<cv::Point2d>>areaPoses,
	vector<Mat>& dCs, vector<Mat>& precisions,
	vector<double> angleIncertitudes)
{
	if (areaGrads.size() != areaPoses.size())//意外错误检测
		return false;

	//对数据统一进行了线性归一化,计算统一的H矩阵
	Mat a, b, c;
	for (uint i = 0; i < areaGrads.size(); i++)
	{
		Mat ctemp;
		Mat areaGradmat = Mat(areaGrads[i]).reshape(1);
		Mat areaPosmat = Mat(areaPoses[i]).reshape(1);
		a.push_back(areaGradmat.col(0));
		b.push_back(areaGradmat.col(1));
		Mat multitemp = areaGradmat.mul(areaPosmat);
		addWeighted(multitemp.col(0), -1, multitemp.col(1), -1, 0, ctemp);
		c.push_back(ctemp);
	}
	Mat M = Mat(a.rows, 2, CV_64F);
	Mat tempb = -1 * b;
	tempb.copyTo(M.col(0));
	a.copyTo(M.col(1));
	Mat B = -1 * c;
	Mat mpts, Lnorm, Lnormt, Minvert;
	if (!solve(M, B, mpts, DECOMP_SVD))
		return false;
	Mat H = Mat::eye(3, 3, CV_64F);
	H.at<double>(0, 2) = mpts.at<double>(0, 0);
	H.at<double>(1, 2) = mpts.at<double>(1, 0);
	//// 构造完毕H

	vector<Mat> AAVec, BBVec, aVec, cVec;
	for (uint i = 0; i < areaGrads.size(); i++)
	{
		Mat a, b, c;
		Mat areaGradmat = Mat(areaGrads[i]).reshape(1);
		Mat areaPosmat = Mat(areaPoses[i]).reshape(1);
		a = areaGradmat.col(0);
		b = areaGradmat.col(1);
		Mat multitemp = areaGradmat.mul(areaPosmat);
		addWeighted(multitemp.col(0), -1, multitemp.col(1), -1, 0, c);

		//构造[a,b,c]
		Mat abc = Mat(a.rows, 3, CV_64F);
		a.copyTo(abc.col(0));
		b.copyTo(abc.col(1));
		c.copyTo(abc.col(2));

		//得到归一化后的a,b和c
		Lnorm = H.t()*abc.t();
		Lnormt = Lnorm.t();
		a = Lnormt.col(0).clone();
		b = Lnormt.col(1).clone();
		c = Lnormt.col(2).clone();
		Mat AA = Mat(5, 5, CV_64F);
		Mat BB = Mat(5, 1, CV_64F);
		Mat a2 = a.mul(a);
		Mat ab = a.mul(b);
		Mat b2 = b.mul(b);
		Mat ac = a.mul(c);
		Mat bc = b.mul(c);
		Mat c2 = c.mul(c);
		aVec.push_back(a);//存储a和c,用于误差计算
		cVec.push_back(c);

		//solution par least-square
		//求AA
		Mat aaaa = a2.mul(a2); Mat aaab = a2.mul(ab); Mat aabb = a2.mul(b2); Mat aaac = a2.mul(ac); Mat aabc = a2.mul(bc);
		Mat abab = ab.mul(ab); Mat abbb = ab.mul(b2); Mat abac = ab.mul(ac); Mat abbc = ab.mul(bc);
		Mat bbbb = b2.mul(b2); Mat bbac = b2.mul(ac); Mat bbbc = b2.mul(bc);
		Mat acac = ac.mul(ac); Mat acbc = ac.mul(bc);
		Mat bcbc = bc.mul(bc);
		AA.at<double>(0, 0) = sum(aaaa).val[0]; AA.at<double>(0, 1) = sum(aaab).val[0]; AA.at<double>(0, 2) = sum(aabb).val[0]; AA.at<double>(0, 3) = sum(aaac).val[0]; AA.at<double>(0, 4) = sum(aabc).val[0];
		AA.at<double>(1, 0) = sum(aaab).val[0]; AA.at<double>(1, 1) = sum(abab).val[0]; AA.at<double>(1, 2) = sum(abbb).val[0]; AA.at<double>(1, 3) = sum(abac).val[0]; AA.at<double>(1, 4) = sum(abbc).val[0];
		AA.at<double>(2, 0) = sum(aabb).val[0]; AA.at<double>(2, 1) = sum(abbb).val[0]; AA.at<double>(2, 2) = sum(bbbb).val[0]; AA.at<double>(2, 3) = sum(bbac).val[0]; AA.at<double>(2, 4) = sum(bbbc).val[0];
		AA.at<double>(3, 0) = sum(aaac).val[0]; AA.at<double>(3, 1) = sum(abac).val[0]; AA.at<double>(3, 2) = sum(bbac).val[0]; AA.at<double>(3, 3) = sum(acac).val[0]; AA.at<double>(3, 4) = sum(acbc).val[0];
		AA.at<double>(4, 0) = sum(aabc).val[0]; AA.at<double>(4, 1) = sum(abbc).val[0]; AA.at<double>(4, 2) = sum(bbbc).val[0]; AA.at<double>(4, 3) = sum(acbc).val[0]; AA.at<double>(4, 4) = sum(bcbc).val[0];

		//求BB
		Mat _ccaa = -1 * (c2.mul(a2)); Mat _ccab = -1 * (c2.mul(ab)); Mat _ccbb = -1 * (c2.mul(b2)); Mat _ccac = -1 * (c2.mul(ac)); Mat _ccbc = -1 * (c2.mul(bc));
		BB.at<double>(0, 0) = sum(_ccaa).val[0]; BB.at<double>(1, 0) = sum(_ccab).val[0]; BB.at<double>(2, 0) = sum(_ccbb).val[0]; BB.at<double>(3, 0) = sum(_ccac).val[0]; BB.at<double>(4, 0) = sum(_ccbc).val[0];
		if (determinant(AA) < 10e-10)
		{
			//是否没有必要做下面工作，直接return false'
			//            dC = Mat::ones(3,3,CV_64F);
			//            dC = -1*dC;
			//            precision.at<double>(0,0) = -1;
			//            angleIncertitude = -1;
			return false;
		}
		AAVec.push_back(AA);
		BBVec.push_back(BB);
	}

	//求AAU和BBU
	Mat AAU;// = Mat(5*areaGrads.size(),3*areaGrads.size()+2,CV_64F);
	Mat BBU;// = Mat(5*areaGrads.size(),1,CV_64F);
	for (uint j = 0; j < areaGrads.size(); j++)//5行5行的压入到AAU和BBU中
	{
		Mat AAURow, AAURowT;
		for (uint k = 0; k < 3 * areaGrads.size(); k++)//先压入AA的前三列到AAU中
		{
			if (j != k%areaGrads.size())
			{
				Mat O = Mat::zeros(1, 5, CV_64F);
				AAURowT.push_back(O);
			}
			else
			{
				Mat tempM = Mat::zeros(5, 5, CV_64F);
				AAVec[j].copyTo(tempM);
				tempM = tempM.t();
				AAURowT.push_back((tempM.row(k / areaGrads.size())));
			}
		}
		for (uint k = 0; k < 2; k++)//再压入AA的后两列到AAU中
		{
			Mat tempM = Mat::zeros(5, 5, CV_64F);
			AAVec[j].copyTo(tempM);
			tempM = tempM.t();
			AAURowT.push_back((tempM.row(k + 3)));
		}
		AAURow = AAURowT.t();//转置并压入到AAU中
		AAU.push_back(AAURow);
		BBU.push_back(BBVec[j]);//直接将相应的BB压入到BBU中
	}

	//解AAU*THITAU=BBU;
	//    cout<<AAU<<endl;
	//    cout<<BBU<<endl;
	Mat solu;// = Mat(3*areaGrads.size()+2,1,CV_64F);
	if (!solve(AAU, BBU, solu, DECOMP_SVD))
		return false;

	//先拆解
	vector<Mat> dCnormVec;
	vector<Mat> solVec;
	for (uint i = 0; i < areaGrads.size(); i++)
	{
		Mat dCnorm = Mat::ones(3, 3, CV_64F);
		Mat sol = Mat::ones(5, 1, CV_64F);
		dCnorm.at<double>(0, 2) = solu.at<double>(3 * areaGrads.size(), 0) / 2;
		dCnorm.at<double>(2, 0) = solu.at<double>(3 * areaGrads.size(), 0) / 2;
		dCnorm.at<double>(1, 2) = solu.at<double>(3 * areaGrads.size() + 1, 0) / 2;
		dCnorm.at<double>(2, 1) = solu.at<double>(3 * areaGrads.size() + 1, 0) / 2;
		sol.at<double>(3, 0) = solu.at<double>(3 * areaGrads.size(), 0);
		sol.at<double>(4, 0) = solu.at<double>(3 * areaGrads.size() + 1, 0);
		dCnormVec.push_back(dCnorm);
		solVec.push_back(sol);
	}
	for (uint i = 0; i < 3 * areaGrads.size(); i++)
	{

		uint j = 0;//对应于A,B,C
		if (0 == i%areaGrads.size())
			j++;
		if (j == 1)//A
		{
			dCnormVec[i%areaGrads.size()].at<double>(0, 0) = solu.at<double>(i, 0);
			solVec[i%areaGrads.size()].at<double>(0, 0) = solu.at<double>(i, 0);
		}
		else if (j == 2)//B
		{
			dCnormVec[i%areaGrads.size()].at<double>(0, 1) = solu.at<double>(i, 0) / 2;
			dCnormVec[i%areaGrads.size()].at<double>(1, 0) = solu.at<double>(i, 0) / 2;
			solVec[i%areaGrads.size()].at<double>(1, 0) = solu.at<double>(i, 0);
		}
		else if (j == 3)//C
		{
			dCnormVec[i%areaGrads.size()].at<double>(1, 1) = solu.at<double>(i, 0);
			solVec[i%areaGrads.size()].at<double>(2, 0) = solu.at<double>(i, 0);
		}
	}

	//denormalisation
	for (uint i = 0; i < areaGrads.size(); i++)
	{
		Mat dC;
		dC = H*dCnormVec[i] * H.t();
		dCs.push_back(dC);
	}

	//误差估计
	for (uint i = 0; i < areaGrads.size(); i++)
	{
		Mat w, u, vt, precision;
		Mat ss = solVec[i].t();
		Mat c2 = cVec[i].mul(cVec[i]);
		Mat cccc = c2.mul(c2);
		double angleIncertitude;
		double BIB = sum(cccc).val[0];
		Mat R = (ss*AAVec[i] * solVec[i] - 2 * ss*BBVec[i] + BIB) / (aVec[i].rows - 5);
		double RmatValue = R.at<double>(0, 0);
		Mat cvar2_constantVariance = RmatValue*AAVec[i].inv();
		double vD = cvar2_constantVariance.at<double>(3, 3);
		double vDE = cvar2_constantVariance.at<double>(3, 4);
		double vE = cvar2_constantVariance.at<double>(4, 4);
		Mat errorMatrics = Mat(2, 2, CV_64F);
		errorMatrics.at<double>(0, 0) = cvar2_constantVariance.at<double>(3, 3);
		errorMatrics.at<double>(0, 1) = cvar2_constantVariance.at<double>(3, 4);
		errorMatrics.at<double>(1, 0) = cvar2_constantVariance.at<double>(4, 3);
		errorMatrics.at<double>(1, 1) = cvar2_constantVariance.at<double>(4, 4);
		SVD::compute(errorMatrics, w, u, vt);
		Mat diagresult;
		sqrt(w, diagresult);
		diagresult = diagresult / 4;
		precision = diagresult.t();
		precisions.push_back(precision);
		angleIncertitude = atan2(vt.at<double>(1, 1), vt.at<double>(0, 1));
		angleIncertitudes.push_back(angleIncertitude);
	}

	return true;
}

double  CoreAlgorithm::distancePoints(Point2f Pnt_1, Point2f Pnt_2)
{
	double distance = sqrt(pow(Pnt_1.x - Pnt_2.x, 2) + pow(Pnt_1.y - Pnt_2.y, 2));
	return distance;
}

// createT7Img.cpp : 定义控制台应用程序的入口点。

//radius  圆半径，（单位像素）；width,生成模板图像大小，正方形
Mat CoreAlgorithm::circleImg(int radius, int width)
{
	Mat img = Mat(width, width, CV_8UC1, Scalar(255));
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			if (sqrt(pow(i - (width - 1) / 2.0, 2) + pow(j - (width - 1) / 2.0, 2))>radius)
				continue;
			else
				img.at<uchar>(i, j) = (uchar)0;
		}
	}
	return img;
}

bool CoreAlgorithm::copyCircleToImg(const Mat circleImg, Point2i offset, Point2i location, Mat &writeImg)
{
	location.x += offset.x;
	location.y += offset.y;
	if (location.x - (circleImg.cols - 1) / 2.0<0 || location.y - (circleImg.rows - 1) / 2.0<0 ||
		location.x + (circleImg.cols - 1) / 2.0>writeImg.cols || location.y + (circleImg.rows - 1) / 2.0>writeImg.rows)
		return false;
	Point2i circleImgstart = Point2i(location.x - (circleImg.cols - 1) / 2, location.y - (circleImg.rows - 1) / 2);
	for (int i = 0; i < circleImg.rows; i++)
	{
		for (int j = 0; j < circleImg.cols; j++)
		{
			writeImg.at<uchar>(circleImgstart.y + i, circleImgstart.x + j) = circleImg.at<uchar>(i, j);
		}
	}
	return true;

}

Mat CoreAlgorithm::DrawEllipse(Mat img, double EllipseCenter_x, double EllipseCenter_y, double EllipseLong_axis, double EllipseShort_axis, double angle, int ksizeWidth)
{

	int thickness = -2;
	int lineType = 8;
	ellipse(img,
		Point(EllipseCenter_x, EllipseCenter_y),
		Size(EllipseLong_axis, EllipseShort_axis),   ////ellipse()函数中参数轴长应该是长短轴的一半，此处将对应的参数除以二，则我们输入即可认为是长短轴轴长。
		angle,
		0,
		360,
		Scalar(0,255,0),
		thickness,
		lineType);
	//Mat out;
	//blur(img, out, Size(ksizeWidth, ksizeWidth));
	return img;
}







bool CoreAlgorithm::findPntsWithTags(vector<Point2f>& centerPnt, vector<float>& longaxisRadius, vector<TagPoint2f>& TagPnts, float &firstFeaturelength, const double gamma)
{

 	if (centerPnt.size() < 15)
		return false;
	if (centerPnt.size() != longaxisRadius.size())
		return false;
	int pntsSize = centerPnt.size();
	Mat distanceMatrix1 = Mat::zeros(pntsSize, pntsSize, CV_64F);
	Mat distanceMatrix = Mat::zeros(pntsSize, pntsSize, CV_64F);
	///////STEP-1：计算椭圆中心之间的距离并除以主椭圆长轴半径，将其存储在矩阵中
	for (int i = 0; i < pntsSize; i++)
	{
		for (int j = i + 1; j < pntsSize; j++)
		{
			distanceMatrix1.at<double>(i, j) = CoreAlgorithm::distancePoints(centerPnt[i], centerPnt[j]) / (longaxisRadius[i] * gamma);
		}
	}
	add(distanceMatrix1, distanceMatrix1.t(), distanceMatrix);
	////////分析数据矩阵，得出每个点临近点的序号和个数
	vector<pair<int, vector<int>>>   threeVec, fourVec, fiveVec;////此处pair<int, vector<int>>可以看成是一个结构体，通过first和second来调用里面的数据。
	for (int i = 0; i < distanceMatrix.rows; i++)
	{
		pair<int, vector<int>>  tempData;
		tempData.first = i;
		for (int j = 0; j < distanceMatrix.cols; j++)
		{
			if (distanceMatrix.at<double>(i, j)>0.9&&distanceMatrix.at<double>(i, j)<1.2
				&&longaxisRadius[i] / longaxisRadius[j]>0.8&&longaxisRadius[i] / longaxisRadius[j] < 1.2)
				tempData.second.push_back(j);
		}
		switch (tempData.second.size())
		{
		case 3:
			threeVec.push_back(tempData);
			continue;
		case 4:
			fourVec.push_back(tempData);
			continue;
		case 5:
			fiveVec.push_back(tempData);
			continue;
		default:
			continue;
		}
	}
	/////根据邻近点个数分布情况，进行分类。
	//////识别1号点
	if (fourVec.size() != 1)
		return false;
	TagPoint2f firstPoint;
	firstPoint[0] = TAG1;
	firstPoint[1] = centerPnt[fourVec[0].first].x;
	firstPoint[2] = centerPnt[fourVec[0].first].y;
	TagPnts.push_back(firstPoint);
	firstFeaturelength = longaxisRadius[fourVec[0].first];
	
	//////识别2号点
	if (threeVec.size() != 1)
		return false;
	TagPoint2f secondPoint;
	secondPoint[0] = TAG2;
	secondPoint[1] = centerPnt[threeVec[0].first].x;
	secondPoint[2] = centerPnt[threeVec[0].first].y;
	TagPnts.push_back(secondPoint);
	
	//////识别3号点
	if (fiveVec.size() != 1)
		return false;
	TagPoint2f thirdPoint;
	thirdPoint[0] = TAG3;
	thirdPoint[1] = centerPnt[fiveVec[0].first].x;
	thirdPoint[2] = centerPnt[fiveVec[0].first].y;
	TagPnts.push_back(thirdPoint);
	
	//////识别4号点
	TagPoint2f  fourthPoint;
	for (size_t i = 0; i < threeVec[0].second.size(); i++)
	{
		size_t j = threeVec[0].second[i];
		if (abs(abs((centerPnt[j].y - firstPoint[2]) / (centerPnt[j].x - firstPoint[1])) - abs((firstPoint[2] - thirdPoint[2]) / (firstPoint[1] - thirdPoint[1]))) <0.02)
		{
			fourthPoint[0] = TAG4;
			fourthPoint[1] = centerPnt[j].x;
			fourthPoint[2] = centerPnt[j].y;
			TagPnts.push_back(fourthPoint);
			
		}
		else
		{
			continue;
		}
	}
	///////识别5号点
	TagPoint2f fifthPoint, sixthPoint;
	vector<int>  fiveSixData;
	for (size_t i = 0; i < centerPnt.size(); i++)
	{
		if (abs(abs((centerPnt[i].y - firstPoint[2]) / (centerPnt[i].x - firstPoint[1])) - abs((firstPoint[2] - thirdPoint[2]) / (firstPoint[1] - thirdPoint[1])) <0.02))
		{
			for (size_t j = 0; j < fourVec[0].second.size(); j++)
			{
				if (i == fourVec[0].second[j])
				{
					//////根据向量P1P4，P1P5，P1P6,计算P1P4与P1P5，P1P4与P1P6的向量积，根据正负来判别5和6号标志点
					Vec2f  NormP1P4, NormPiP1;
					NormP1P4[0] = fourthPoint[1] - firstPoint[1];
					NormP1P4[1] = fifthPoint[2] - fifthPoint[2];
					NormPiP1[0] = centerPnt[i].x - firstPoint[1];
					NormPiP1[1] = centerPnt[i].y - firstPoint[2];
					double dotResult = NormP1P4.dot(NormPiP1);
					if (dotResult < 0)
					{
						fifthPoint[0] = TAG5;
						fifthPoint[1] = centerPnt[i].x;
						fifthPoint[2] = centerPnt[i].y;
						TagPnts.push_back(fifthPoint);
						fiveSixData.push_back(i);
						
						
					}
					else
					{
						sixthPoint[0] = TAG6;
						sixthPoint[1] = centerPnt[i].x;
						sixthPoint[2] = centerPnt[i].y;
						TagPnts.push_back(sixthPoint);
						fiveSixData.push_back(i);
						
					}
				}
				else
				{
					continue;
				}
			}
		}
		else
		{
			continue;
		}

	}
	///////识别标志点7,8号
	vector<pair<int, Vec2f>>  NormP1P;
	pair<int,Vec2f>  NormP1P78;
	for (size_t i = 0; i < fourVec[0].second.size(); i++)
	{
		if (fourVec[0].second[i]!= fiveSixData[0] && fourVec[0].second[i] != fiveSixData[1])
			{
				NormP1P78.first = fourVec[0].second[i];
				NormP1P78.second[0] = centerPnt[fourVec[0].second[i]].x - firstPoint[1];
				NormP1P78.second[1] = centerPnt[fourVec[0].second[i]].y - firstPoint[2];
				double absValue = sqrt(pow(NormP1P78.second[0], 2) + pow(NormP1P78.second[1], 2));
				NormP1P78.second[0] = NormP1P78.second[0] / absValue;
				NormP1P78.second[1] = NormP1P78.second[1] / absValue;
				NormP1P.push_back(NormP1P78);
			}
			else
			{
				continue;
			}
	}
	TagPoint2f seventhPoint;
	TagPoint2f eighthPoint;
	double value = NormP1P[0].second.dot(NormP1P[1].second);
	if (NormP1P[0].second.dot(NormP1P[1].second) <= -0.9 && NormP1P[0].second.dot(NormP1P[1].second)>=-1)
	{
		Vec2f NormP1P2;
		NormP1P2[0] = secondPoint[1] - firstPoint[1];
		NormP1P2[1] = secondPoint[2] - firstPoint[2];
		if (NormP1P[0].second.dot(NormP1P2)>0)
		{
			seventhPoint[0] = TAG7;
			seventhPoint[1] = centerPnt[NormP1P[0].first].x;
			seventhPoint[2] = centerPnt[NormP1P[0].first].y;
			
			eighthPoint[0] = TAG8;
			eighthPoint[1] = centerPnt[NormP1P[1].first].x;
			eighthPoint[2] = centerPnt[NormP1P[1].first].y;
			
		}
		else
		{
			eighthPoint[0] = TAG8;
			eighthPoint[1] = centerPnt[NormP1P[0].first].x;
			eighthPoint[2] = centerPnt[NormP1P[0].first].y;
		
			seventhPoint[0] = TAG7;
			seventhPoint[1] = centerPnt[NormP1P[1].first].x;
			seventhPoint[2] = centerPnt[NormP1P[1].first].y;
		
		}
	}
	else
	{
		return false;
	}

	//////////识别标志点9,10,11号
	TagPoint2f  ninthPoint;
	TagPoint2f  tenthPoint;
	TagPoint2f  eleventhPoint;
	vector<int> eleventhData;
	for (size_t i = 0; i < centerPnt.size(); i++)
	{
		if (abs(abs((centerPnt[i].y - secondPoint[2]) / (centerPnt[i].x - secondPoint[1])) - abs((secondPoint[2] - seventhPoint[2]) / (secondPoint[1] - seventhPoint[1])) <0.02))
		{

			Vec2f  NormP7P2;
			NormP7P2[0] = secondPoint[1] - seventhPoint[1];
			NormP7P2[1] = secondPoint[2] - seventhPoint[2];
			for (size_t j = 0; j < threeVec[0].second.size(); j++)
			{
				if (i == threeVec[0].second[j])
				{
					Vec2f NormP2Pi;
					NormP2Pi[0] = centerPnt[i].x - secondPoint[1];
					NormP2Pi[1] = centerPnt[i].y - secondPoint[2];
					if (NormP2Pi.dot(NormP7P2)<0)
					{
						ninthPoint[0] = TAG9;
						ninthPoint[1] = centerPnt[i].x;
						ninthPoint[2] = centerPnt[i].y;
						

					}
					else
					{
						tenthPoint[0] = TAG10;
						tenthPoint[1] = centerPnt[i].x;
						tenthPoint[2] = centerPnt[i].y;
						
					}
				}
				else
				{
					for (size_t k = 0; k < fiveVec[0].second.size(); k++)
					{
						if (i == fiveVec[0].second[k])
						{
							eleventhPoint[0] = TAG11;
							eleventhPoint[1] = centerPnt[i].x;
							eleventhPoint[2] = centerPnt[i].y;
							eleventhData.push_back(i);
							
						}
						else
						{
							continue;
						}


					}
				}
			}

		}
		else
		{
			continue;
		}
	}

	///////识别标志点12,13,14,15号
	TagPoint2f twelfthPoint;
	TagPoint2f thirteenthPoint;
	TagPoint2f fourteenthPoint;
	TagPoint2f fifteenthPoint;
	Vec2f NormP3P11, NormP3P4;
	vector<pair<int, Vec2f>>  A, B;
	pair<int, Vec2f> a, b;
	Vec2f NormP3Pi;
	NormP3P11[0] = eleventhPoint[1] - thirdPoint[1];
	NormP3P11[1] = eleventhPoint[2] - thirdPoint[2];
	NormP3P4[0] = fourthPoint[1] - thirdPoint[1];
	NormP3P4[1] = fourthPoint[2] - thirdPoint[2];
	for (size_t i = 0; i < fiveVec[0].second.size(); i++)
	{
		if (fiveVec[0].second[i] != eleventhData[0])
		{
			NormP3Pi[0] = centerPnt[fiveVec[0].second[i]].x - thirdPoint[1];
			NormP3Pi[1] = centerPnt[fiveVec[0].second[i]].y - thirdPoint[2];
			if (NormP3Pi.dot(NormP3P11) > 0)
			{
				a.first = fiveVec[0].second[i];
				a.second=NormP3Pi;
				A.push_back(a);
			}
			else
			{
				b.first = fiveVec[0].second[i];
				b.second=NormP3Pi;
				B.push_back(b);
			}
		}
		else
		{
			continue;
		}

	}
	/////////识别标志点12,13号
	for (size_t i = 0; i < A.size(); i++)
	{
		if (A[i].second.dot(NormP3P4)>0)
		{
			twelfthPoint[0] = TAG12;
			twelfthPoint[1] = centerPnt[A[i].first].x;
			twelfthPoint[2] = centerPnt[A[i].first].y;
			
		}
		else
		{
			thirteenthPoint[0] = TAG13;
			thirteenthPoint[1] = centerPnt[A[i].first].x;
			thirteenthPoint[2] = centerPnt[A[i].first].y;
			
		}

	}
	///////识别标志点14,15号
	for (size_t i = 0; i < B.size(); i++)
	{
		if (B[i].second.dot(NormP3P4)>0)
		{
			fourteenthPoint[0] = TAG14;
			fourteenthPoint[1] = centerPnt[B[i].first].x;
			fourteenthPoint[2] = centerPnt[B[i].first].y;
		}
		else
		{
			fifteenthPoint[0] = TAG15;
			fifteenthPoint[1] = centerPnt[B[i].first].x;
			fifteenthPoint[2] = centerPnt[B[i].first].y;
		}

	}
	TagPnts.push_back(seventhPoint);
	TagPnts.push_back(eighthPoint);
	TagPnts.push_back(ninthPoint);
	TagPnts.push_back(tenthPoint);
	TagPnts.push_back(eleventhPoint);
	TagPnts.push_back(twelfthPoint);
	TagPnts.push_back(thirteenthPoint);
	TagPnts.push_back(fourteenthPoint);
	TagPnts.push_back(fifteenthPoint);
}