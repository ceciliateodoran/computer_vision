#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include "opencv2/videoio.hpp"

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <filesystem>
#define _USE_MATH_DEFINES
#include <math.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;
//using namespace cv::sfm;
namespace fs = std::filesystem;

//CALIBRATION FUNCTION
const char* liveCaptureHelp =
"When the live video from camera is used as input, the following hot-keys may be used:\n"
"  <ESC>, 'q' - quit the program\n"
"  'g' - start capturing images\n"
"  'u' - switch undistortion on/off\n";

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2, MEASURING = 3 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

static double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
	corners.resize(0);

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float(j * squareSize),
					float(i * squareSize), 0));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
					float(i * squareSize), 0));
		break;

	default:
		CV_Error(Error::StsBadArg, "Unknown pattern type\n");
	}
}

vector<float> reprojErrs;
double totalAvgErr = 0;

static bool runCalibration(vector<vector<Point2f> > imagePoints,
	Size imageSize, Size boardSize, Pattern patternType,
	float squareSize, float aspectRatio,
	float grid_width, bool release_object,
	int flags, Mat& cameraMatrix, Mat& distCoeffs,
	vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs,
	vector<Point3f>& newObjPoints,
	double& totalAvgErr)
{
	if (flags & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
	objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
	newObjPoints = objectPoints[0];

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms;
	int iFixedPoint = -1;
	if (release_object)
		iFixedPoint = boardSize.width - 1;
	rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
		cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
		flags | CALIB_USE_LU);
	printf("RMS error reported by calibrateCamera: %g\n", rms);

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	if (release_object) {
		cout << "New board corners: " << endl;
		cout << newObjPoints[0] << endl;
		cout << newObjPoints[boardSize.width - 1] << endl;
		cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << endl;
		cout << newObjPoints.back() << endl;
	}

	objectPoints.clear();
	objectPoints.resize(imagePoints.size(), newObjPoints);
	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

static bool loadCameraParams(const string& filename, Size& imageSize, Size& boardSize, Mat& cameraMatrix, Mat& distCoeffs, float& squareSize, double& totalAvgErr) {
	FileStorage fs(filename, FileStorage::READ);

	imageSize.width = fs["image_width"];
	imageSize.height = fs["image_height"];
	boardSize.width = fs["board_width"];
	boardSize.height = fs["board_height"];
	fs["square_size"] >> squareSize;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;

	totalAvgErr = fs["avg_reprojection_error"];
	return (fs.isOpened());
}

static void saveCameraParams(const string& filename,
	Size imageSize, Size boardSize,
	float squareSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Point3f>& newObjPoints,
	double totalAvgErr)
{
	FileStorage fs(filename, FileStorage::WRITE);

	time_t tt;
	time(&tt);
	struct tm* t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flags & CALIB_FIX_ASPECT_RATIO)
		fs << "aspectRatio" << aspectRatio;

	if (flags != 0)
	{
		snprintf(buf, sizeof(buf), "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
		//cvWriteComment( *fs, buf, 0 );
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		//cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
		fs << "extrinsic_parameters" << bigmat;
	}

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}

	if (!newObjPoints.empty())
	{
		fs << "grid_points" << newObjPoints;
	}
}

static void createPlane(vector<cv::Point2f>& projectedPlanes,
	std::vector<Point>& Xplane, std::vector<Point>& Yplane, std::vector<Point>& Zplane) {

	Xplane.push_back({ Point(projectedPlanes.at(0)) });
	Xplane.push_back({ Point(projectedPlanes.at(1)) });
	Xplane.push_back({ Point(projectedPlanes.at(4)) });
	Xplane.push_back({ Point(projectedPlanes.at(3)) });

	Yplane.push_back({ Point(projectedPlanes.at(0)) });
	Yplane.push_back({ Point(projectedPlanes.at(2)) });
	Yplane.push_back({ Point(projectedPlanes.at(5)) });
	Yplane.push_back({ Point(projectedPlanes.at(3)) });

	Zplane.push_back({ Point(projectedPlanes.at(0)) });
	Zplane.push_back({ Point(projectedPlanes.at(1)) });
	Zplane.push_back({ Point(projectedPlanes.at(6)) });
	Zplane.push_back({ Point(projectedPlanes.at(2)) });

}

static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	size_t dir_pos = filename.rfind('/');
	if (dir_pos == string::npos)
		dir_pos = filename.rfind('\\');
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		string fname = (string)*it;
		if (dir_pos != string::npos)
		{
			string fpath = samples::findFile(filename.substr(0, dir_pos + 1) + fname, false);
			if (fpath.empty())
			{
				fpath = samples::findFile(fname);
			}
			fname = fpath;
		}
		else
		{
			fname = samples::findFile(fname);
		}
		l.push_back(fname);
	}
	return true;
}

static bool runAndSave(const string& outputFilename,
	const vector<vector<Point2f> >& imagePoints,
	Size imageSize, Size boardSize, Pattern patternType, float squareSize,
	float grid_width, bool release_object,
	float aspectRatio, int flags, Mat& cameraMatrix,
	Mat& distCoeffs, bool writeExtrinsics, bool writePoints, bool writeGrid)
{
	vector<Mat> rvecs, tvecs;
	vector<Point3f> newObjPoints;

	bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
		aspectRatio, grid_width, release_object, flags, cameraMatrix, distCoeffs,
		rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
	printf("%s. avg reprojection error = %.7f\n",
		ok ? "Calibration succeeded" : "Calibration failed",
		totalAvgErr);

	if (ok)
		saveCameraParams(outputFilename, imageSize,
			boardSize, squareSize, aspectRatio,
			flags, cameraMatrix, distCoeffs,
			writeExtrinsics ? rvecs : vector<Mat>(),
			writeExtrinsics ? tvecs : vector<Mat>(),
			writeExtrinsics ? reprojErrs : vector<float>(),
			writePoints ? imagePoints : vector<vector<Point2f> >(),
			writeGrid ? newObjPoints : vector<Point3f>(),
			totalAvgErr);
	return ok;
}
//END CALIBRATION FUNCTION

int main(int argc, char* argv[])
{
	//START CALIBRATION
	Size boardSize, imageSize;
	float squareSize, aspectRatio = 1;
	Mat cameraMatrix, distCoeffs;
	string outputFilename;
	string inputFilename = "";
	int i, nframes;
	bool writeExtrinsics, writePoints;
	bool undistortImage = false;
	int flags = 0;
	VideoCapture capture;
	bool flipVertical;
	bool showUndistorted;
	bool videofile;
	int delay;
	clock_t prevTimestamp = 0;
	int mode = DETECTION;
	int cameraId = 1;
	vector<vector<Point2f> > imagePoints;
	vector<string> imageList;
	Pattern pattern = CHESSBOARD;

	cv::CommandLineParser parser(argc, argv,
		"{help ||}{w||}{h||}{pt|chessboard|}{n|10|}{d|1000|}{s|1|}{o|out_camera_data.yml|}"
		"{op||}{oe||}{zt||}{a||}{p||}{v||}{V||}{su||}"
		"{oo||}{ws|11|}{dt||}"
		"{fx||}{fy||}{cx||}{cy||}"
		"{imshow-scale|1|}{enable-k3|0|}"
		"{@input_data|0|}"
		"{i|out_camera_data.yml|}");
	boardSize.width = parser.get<int>("w");
	boardSize.height = parser.get<int>("h");
	if (parser.has("pt"))
	{
		string val = parser.get<string>("pt");
		if (val == "circles")
			pattern = CIRCLES_GRID;
		else if (val == "acircles")
			pattern = ASYMMETRIC_CIRCLES_GRID;
		else if (val == "chessboard")
			pattern = CHESSBOARD;
		else
			return fprintf(stderr, "Invalid pattern type: must be chessboard or circles\n"), -1;
	}
	squareSize = parser.get<float>("s");
	nframes = parser.get<int>("n");
	delay = parser.get<int>("d");
	writePoints = parser.has("op");
	writeExtrinsics = parser.has("oe");
	bool writeGrid = parser.has("oo");
	if (parser.has("a")) {
		flags |= CALIB_FIX_ASPECT_RATIO;
		aspectRatio = parser.get<float>("a");
	}
	if (parser.has("zt"))
		flags |= CALIB_ZERO_TANGENT_DIST;
	if (parser.has("p"))
		flags |= CALIB_FIX_PRINCIPAL_POINT;
	flipVertical = parser.has("v");
	videofile = parser.has("V");
	if (parser.has("o"))
		outputFilename = parser.get<string>("o");
	showUndistorted = parser.has("su");
	if (isdigit(parser.get<string>("@input_data")[0]))
		cameraId = parser.get<int>("@input_data");
	else
		inputFilename = parser.get<string>("@input_data");
	int winSize = parser.get<int>("ws");
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (parser.has("fx") && parser.has("fy") && parser.has("cx") && parser.has("cy"))
	{
		cameraMatrix.at<double>(0, 0) = parser.get<double>("fx");
		cameraMatrix.at<double>(0, 2) = parser.get<double>("cx");
		cameraMatrix.at<double>(1, 1) = parser.get<double>("fy");
		cameraMatrix.at<double>(1, 2) = parser.get<double>("cy");
		flags |= CALIB_USE_INTRINSIC_GUESS;
		std::cout << "Use the following camera matrix as an initial guess:\n" << cameraMatrix << std::endl;
	}

	if (parser.has("i")) {
		std::string loadFilename = parser.get<string>("i");
		struct stat buffer;
		if (stat(loadFilename.c_str(), &buffer) == 0 && loadCameraParams(loadFilename, imageSize, boardSize, cameraMatrix, distCoeffs, squareSize, totalAvgErr)) {
			mode = CALIBRATED;
		}
	}

	int viewScaleFactor = parser.get<int>("imshow-scale");
	bool useK3 = parser.get<bool>("enable-k3");
	std::cout << "Use K3 distortion coefficient? " << useK3 << std::endl;
	if (!useK3)
	{
		flags |= CALIB_FIX_K3;
	}
	float grid_width = squareSize * (boardSize.width - 1);
	bool release_object = false;
	if (parser.has("dt")) {
		grid_width = parser.get<float>("dt");
		release_object = true;
	}
	if (squareSize <= 0)
		return fprintf(stderr, "Invalid board square width\n"), -1;
	if (nframes <= 3)
		return printf("Invalid number of images\n"), -1;
	if (aspectRatio <= 0)
		return printf("Invalid aspect ratio\n"), -1;
	if (delay <= 0)
		return printf("Invalid delay\n"), -1;
	if (boardSize.width <= 0)
		return fprintf(stderr, "Invalid board width\n"), -1;
	if (boardSize.height <= 0)
		return fprintf(stderr, "Invalid board height\n"), -1;

	if (!inputFilename.empty())
	{
		if (!videofile && readStringList(samples::findFile(inputFilename), imageList))
			mode = CAPTURING;
		else
			capture.open(samples::findFileOrKeep(inputFilename));
	}
	else
		capture.open(cameraId);

	if (!capture.isOpened() && imageList.empty())
		return fprintf(stderr, "Could not initialize video (%d) capture\n", cameraId), -2;

	if (!imageList.empty())
		nframes = (int)imageList.size();

	if (capture.isOpened())
		printf("%s", liveCaptureHelp);

	const char* winName = "Image View";
	namedWindow(winName, 1);

	vector<vector<Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0], pattern);
	objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;

	//objectPoints.resize(imagePoints.size(), objectPoints[0]);

	char key = (char)waitKey(capture.isOpened() ? 50 : 500);
	int index = 0;
	float max_re = 0;
	bool fineTuning = false;

	vector<Point2f> pointBufTmp;

	for (i = 0;; i++)
	{
		Mat view, viewGray;
		bool blink = false;

		if (capture.isOpened())
		{
			Mat view0;
			capture >> view0;
			view0.copyTo(view);
		}
		else if (i < (int)imageList.size())
			view = imread(imageList[i], 1);

		if (view.empty())
		{
			if (imagePoints.size() > 0)
				runAndSave(outputFilename, imagePoints, imageSize,
					boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
					flags, cameraMatrix, distCoeffs,
					writeExtrinsics, writePoints, writeGrid);
			break;
		}

		imageSize = view.size();

		if (flipVertical)
			flip(view, view, 0);
		vector<Point2f> pointbuf;
		cvtColor(view, viewGray, COLOR_BGR2GRAY);

		bool found;
		switch (pattern)
		{
		case CHESSBOARD:
			found = findChessboardCorners(view, boardSize, pointbuf,
				CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
			if (found) { pointBufTmp = pointbuf; }
			break;
		case CIRCLES_GRID:
			found = findCirclesGrid(view, boardSize, pointbuf);
			break;
		case ASYMMETRIC_CIRCLES_GRID:
			found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
			break;
		default:
			return fprintf(stderr, "Unknown pattern type\n"), -1;
		}

		// improve the found corners' coordinate accuracy
		if (pattern == CHESSBOARD && found) cornerSubPix(viewGray, pointbuf, Size(winSize, winSize),
			Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

		if (key == ' ' && mode == CAPTURING && found &&
			(!capture.isOpened() || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC))
		{
			imagePoints.push_back(pointbuf);
			prevTimestamp = clock();
			blink = capture.isOpened();
			namespace fs = std::filesystem;
			fs::create_directory("data");
			fs::create_directory("data_processed");

			if (!fineTuning) {
				string fname = cv::format("data/img%04d.png", imagePoints.size());
				string fname2 = cv::format("data_processed/img%04d.png", imagePoints.size());
				imwrite(fname, view); drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
				imwrite(fname2, view);
			}
			else {
				string fname = cv::format("data/img%04d.png", index + 1);
				string fname2 = cv::format("data_processed/img%04d.png", index + 1);
				imwrite(fname, view); drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
				imwrite(fname2, view);
			}
		}

		if (found)
			drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

		if (reprojErrs.size() > 0) {
			vector<float>::iterator it;
			it = max_element(reprojErrs.begin(), reprojErrs.end());
			max_re = *it;
			index = std::distance(reprojErrs.begin(), it);
		}

		string msg, degreesText, distText = mode == CAPTURING ? "100/100" :
			mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;

		if (mode == CALIBRATED) {
			msg = cv::format("Calibrated: re=%.2f max=%.2f[%d]",
				totalAvgErr, max_re, index);
			fineTuning = false;
		}

		if (mode == CAPTURING)
		{
			if (undistortImage)
				msg = cv::format("%d/%d Undist", (int)imagePoints.size(), nframes);
			else
				msg = cv::format("%d/%d", (int)imagePoints.size(), nframes);
		}

		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2.2 * textSize.width - 10, view.rows - 3 * baseLine - 10);

		putText(view, msg, textOrigin, 1, 1,
			mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

		if (blink)
			bitwise_not(view, view);

		if (mode == CALIBRATED && undistortImage)
		{
			Mat temp = view.clone();
			undistort(temp, view, cameraMatrix, distCoeffs);
		}
		if (viewScaleFactor > 1)
		{
			Mat viewScale;
			resize(view, viewScale, Size(), 1.0 / viewScaleFactor, 1.0 / viewScaleFactor, INTER_AREA);
			imshow("Image View", viewScale);
		}
		else
		{
			imshow("Image View", view);
		}

		if (mode == MEASURING) {
			imshow("Image View", view);
		}

		key = (char)waitKey(capture.isOpened() ? 50 : 500);

		if ((key == 'd') && mode == CALIBRATED && !imagePoints.empty()) {
			cout << "Max el:" << index << endl;
			imagePoints.erase(imagePoints.begin() + index);
			string path_pro = cv::format("data_processed/img%04d.png", index + 1);
			string path = cv::format("data/img%04d.png", index + 1);
			remove(path.c_str());
			remove(path_pro.c_str());
			fineTuning = true;
			mode = CAPTURING;
		}

		if (key == 27 || key == 'q' || key == 'Q')
			break;

		if (key == 'u' && mode == CALIBRATED)
			undistortImage = !undistortImage;

		if (capture.isOpened() && key == 'g')
		{
			mode = CAPTURING;
			imagePoints.clear();
		}


		if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes)
		{
			if (runAndSave(outputFilename, imagePoints, imageSize,
				boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
				flags, cameraMatrix, distCoeffs,
				writeExtrinsics, writePoints, writeGrid))
				mode = CALIBRATED;
			else
				mode = DETECTION;
			if (!capture.isOpened())
				break;
		}

		if (!capture.isOpened() && showUndistorted)
		{
			Mat view, rview, map1, map2;
			initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
				getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
				imageSize, CV_16SC2, map1, map2);

			for (i = 0; i < (int)imageList.size(); i++)
			{
				view = imread(imageList[i], 1);
				if (view.empty())
					continue;
				remap(view, rview, map1, map2, INTER_LINEAR);
				if (viewScaleFactor > 1)
				{
					Mat rviewScale;
					resize(rview, rviewScale, Size(), 1.0 / viewScaleFactor, 1.0 / viewScaleFactor, INTER_AREA);
					imshow("Image View", rviewScale);
				}
				else
				{
					imshow("Image View", rview);
				}
			}
		}
	}
	// END CALIBRATION
	return 0;
}
