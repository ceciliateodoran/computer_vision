#include <iostream>
#include <vector>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

namespace
{
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID, CHARUCOBOARD};

void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch (patternType)
    {
    case CHESSBOARD:
    case CIRCLES_GRID:
        //! [compute-chessboard-object-points]
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float(j*squareSize),
                                          float(i*squareSize), 0));
        //! [compute-chessboard-object-points]
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                          float(i*squareSize), 0));
        break;

    case CHARUCOBOARD:
        for( int i = 0; i < boardSize.height-1; i++ )
            for( int j = 0; j < boardSize.width-1; j++ )
                corners.push_back(Point3f(float(j*squareSize),
                                      float(i*squareSize), 0));
    break;
    default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

// Funzione per ottenere un elenco di immagini da una cartella
vector<string> getImagePaths(const string &folderPath) {
    vector<string> imagePaths;
    struct dirent *entry;
    DIR *dp = opendir(folderPath.c_str());

    if (dp == nullptr) {
        cerr << "Impossibile aprire la cartella: " << folderPath << endl;
        return imagePaths;
    }

    while ((entry = readdir(dp)) != nullptr) {
        // Aggiungi solo i file con estensione .jpg
        string filename = entry->d_name;
        if (filename.find(".jpg") != string::npos || filename.find(".JPG") != string::npos) {
            imagePaths.push_back(folderPath + "/" + filename);
        }
    }
    closedir(dp);
    return imagePaths;
}

void poseEstimationFromCoplanarPoints(const string &folderPath, const string &settingsPath, const Size &boardSize,
                                             const float squareSize)
{
    // Vettori per memorizzare i punti
    vector<vector<cv::Point3f>> objectPoints; // Punti 3D del mondo reale
    vector<vector<cv::Point2f>> imagePoints; // Punti 2D nelle immagini

    // Ottieni i percorsi di tutte le immagini nella cartella
    vector<string> images = getImagePaths(folderPath);
    if (images.empty()) {
        cerr << "Nessuna immagine trovata nella cartella!" << endl;
    }

    // Elenco delle immagini trovate
    for (const auto &imgPath : images) {
        cout << imgPath << endl;
    }

    for (const auto &imagePath : images) {
        Mat img = imread( samples::findFile( imagePath) );
        Mat img_corners = img.clone(), img_pose = img.clone();
        //! [find-chessboard-corners]
        vector<Point2f> corners;
        bool found = findChessboardCorners(img, boardSize, corners);
        //! [find-chessboard-corners]

        if (!found)
        {
            cout << "Cannot find chessboard corners." << endl;
            return;
        }
        drawChessboardCorners(img_corners, boardSize, corners, found);
        imshow("Chessboard corners detection", img_corners);
        waitKey();
    }

    //runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints, grid_width, release_object)

    //! [compute-object-points]
/*    vector<Point3f> objectPoints;
    calcChessboardCorners(boardSize, squareSize, objectPoints);
    vector<Point2f> objectPointsPlanar;
    for (size_t i = 0; i < objectPoints.size(); i++)
    {
        objectPointsPlanar.push_back(Point2f(objectPoints[i].x, objectPoints[i].y));
    }*/
    //! [compute-object-points]

    //calibrateCameraRO

    //! [load-intrinsics]
    /*FileStorage fs( samples::findFile( intrinsicsPath ), FileStorage::READ);
    Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;*/
    //! [load-intrinsics]

    //! [compute-image-points]
/*    vector<Point2f> imagePoints;
    undistortPoints(corners, imagePoints, cameraMatrix, distCoeffs);*/
    //! [compute-image-points]

    //! [estimate-homography]
/*    Mat H = findHomography(objectPointsPlanar, imagePoints);
    cout << "H:\n" << H << endl;*/
    //! [estimate-homography]

    //! [pose-from-homography]
    // Normalization to ensure that ||c1|| = 1
/*    double norm = sqrt(H.at<double>(0,0)*H.at<double>(0,0) +
                       H.at<double>(1,0)*H.at<double>(1,0) +
                       H.at<double>(2,0)*H.at<double>(2,0));

    H /= norm;
    Mat c1  = H.col(0);
    Mat c2  = H.col(1);
    Mat c3 = c1.cross(c2);

    Mat tvec = H.col(2);
    Mat R(3, 3, CV_64F);

    for (int i = 0; i < 3; i++)
    {
        R.at<double>(i,0) = c1.at<double>(i,0);
        R.at<double>(i,1) = c2.at<double>(i,0);
        R.at<double>(i,2) = c3.at<double>(i,0);
    }*/
    //! [pose-from-homography]

    //! [polar-decomposition-of-the-rotation-matrix]
/*    cout << "R (before polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
    Mat_<double> W, U, Vt;
    SVDecomp(R, W, U, Vt);
    R = U*Vt;
    double det = determinant(R);
    if (det < 0)
    {
        Vt.at<double>(2,0) *= -1;
        Vt.at<double>(2,1) *= -1;
        Vt.at<double>(2,2) *= -1;

        R = U*Vt;
    }
    cout << "R (after polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;*/
    //! [polar-decomposition-of-the-rotation-matrix]

    //! [display-pose]
/*    Mat rvec;
    Rodrigues(R, rvec);
    drawFrameAxes(img_pose, cameraMatrix, distCoeffs, rvec, tvec, 2*squareSize);
    imshow("Pose from coplanar points", img_pose);*/
    //waitKey();
    //! [display-pose]
}

static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType,
                    float squareSize, float aspectRatio,
                    float grid_width, int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    vector<Point3f>& newObjPoints,
                    double& totalAvgErr)
{
    if( flags & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
    int offset = patternType != CHARUCOBOARD ? boardSize.width - 1: boardSize.width - 2;
    objectPoints[0][offset].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms;
    int iFixedPoint = -1;
    rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
                            cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
                            flags | CALIB_USE_LU);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       const vector<Point3f>& newObjPoints,
                       double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        snprintf( buf, sizeof(buf), "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if( !newObjPoints.empty() )
    {
        fs << "grid_points" << newObjPoints;
    }
}

static bool runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                float grid_width, float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                   aspectRatio, grid_width, flags, cameraMatrix, distCoeffs,
                   rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
    printf("%s. avg reprojection error = %.7f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    if( ok )
        saveCameraParams( outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, cameraMatrix, distCoeffs,
                         rvecs, tvecs, reprojErrs,
                         imagePoints, newObjPoints,
                         totalAvgErr );
    return ok;
}

static bool loadCameraParams(const string& filename, Size& imageSize, Size& boardSize, Mat& cameraMatrix, Mat& distCoeffs, float& squareSize, double& totalAvgErr) {
	if (!samples::findFile(filename, false).empty()){
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
	return 0;
}

const char* params
    = "{ help h         |       | print usage }"
      "{ images_path           | res | path to chessboard images }"
      "{ outputFilename     | out_camera_data.yml | path to camera intrinsics }"
      "{ width bw       | 9     | chessboard width }"
      "{ height bh      | 6     | chessboard height }"
      "{ square_size    | 0.025 | chessboard square size }";
}

int main(int argc, char *argv[])
{
    float squareSize, grid_width, aspectRatio = 1;
    int flags = 0;
    double totalAvgErr = 0;
    Mat cameraMatrix, distCoeffs;
    string outputFilename;
    string images_path;
    Size imageSize, boardSize;
    Pattern pattern = CHESSBOARD;
    // Vettori per memorizzare i punti
    vector<vector<cv::Point3f>> objectPoints; // Punti 3D del mondo reale
    vector<vector<cv::Point2f>> imagePoints; // Punti 2D nelle immagini

    CommandLineParser parser(argc, argv, params);

    if (parser.has("help"))
    {
        parser.about("Code for homography tutorial.\n"
            "Example 1: pose from homography with coplanar points.\n");
        parser.printMessage();
        return 0;
    }

    boardSize.width = parser.get<int>("width");
    boardSize.height = parser.get<int>("height");
    squareSize = (float) parser.get<double>("square_size");
    images_path = parser.get<String>("images_path");
    outputFilename = parser.get<String>("outputFilename");
    grid_width = squareSize * (boardSize.width - 1);
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    //poseEstimationFromCoplanarPoints(images_path, outputFilename, boardSize, squareSize);

    //CALIBRATION AND PARAMS SAVED
    if(loadCameraParams(outputFilename, imageSize, boardSize, cameraMatrix, distCoeffs, squareSize, totalAvgErr)) {
        cout << "CALIBRATED" << endl;
    } else {
        // Ottieni i percorsi di tutte le immagini nella cartella
        vector<string> images = getImagePaths(images_path);
        if (images.empty()) {
            cerr << "Nessuna immagine trovata nella cartella!" << endl;
        }

        // Elenco delle immagini trovate
        for (const auto &imgPath : images) {
            cout << imgPath << endl;
        }

        for (const auto &imagePath : images) {
            Mat img = imread( samples::findFile( imagePath) );
            imageSize = img.size();
            Mat img_corners = img.clone(), img_pose = img.clone();
            //! [find-chessboard-corners]
            vector<Point2f> corners;
            bool found = findChessboardCorners(img, boardSize, corners);
            //! [find-chessboard-corners]

            if (!found)
            {
                cout << "Cannot find chessboard corners." << endl;
                return 0;
            }else{
                imagePoints.push_back(corners);
            }

            drawChessboardCorners(img_corners, boardSize, corners, found);
            imshow("Chessboard corners detection", img_corners);
            waitKey();
        }

        if( imagePoints.size() > 0 )
            runAndSave(outputFilename, imagePoints, imageSize,
                                   boardSize, pattern, squareSize, grid_width,
                                   aspectRatio, flags, cameraMatrix, distCoeffs);
    }

    return 0;
}
