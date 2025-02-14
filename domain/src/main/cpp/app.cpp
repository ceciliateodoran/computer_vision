#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <thread>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <random>
#include <string>
#include <deque>
#include <vector>
#include <dirent.h>
#include "clustering/dbscan.h"
#include "clustering/ndarray.h"

typedef websocketpp::server<websocketpp::config::asio> Server;
using namespace cv;
using namespace std;
using websocketpp::connection_hdl;
namespace fs = std::filesystem;

enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID, CHARUCOBOARD};

class OpticalFlowTracker {
    struct FeaturePoint {
        Point2f position;      // posizione corrente
        Point2f oldPosition;   // posizione precedente
    };

    struct ClusterData {
        std::vector<cv::Point2f> points;
        cv::Scalar color;
        
        ClusterData(const cv::Scalar& c = cv::Scalar(0,0,0)) : color(c) {}
    };

    // Parametri per Shi-Tomasi
    int maxCorners;
    double qualityLevel;
    double minDistance;
    int blockSize;

    // Parametri per Lucas-Kanade
    cv::TermCriteria criteria;
    cv::Size winSize;
    int maxLevel;

    // Nuovi parametri per il filtraggio delle occorrenze
    float magnitudeTolerance;  // Tolleranza per confrontare i moduli
    float angleTolerance;      // Tolleranza per confrontare gli angoli (in radianti)
    int minOccurrences;        // Numero minimo di occorrenze per considerare un gruppo

    // Background subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2> backSub;

    // Matrici e vettori necessari
    cv::Mat morph;
    cv::Mat ROI_f;
    cv::Mat fgMask;
    cv::Mat old_gray;
    cv::Mat line_mask;
    cv::Mat line_mask_kmeans;
    cv::Mat frame_gray;
    cv::Mat blurred_f;
    std::vector<cv::Point2f> shi_tomasi_keypoints;
    std::vector<cv::Point2f> old_keypoints;
    std::vector<cv::Scalar> colors;
    int frame_counter;

    // Kernels per trasformazioni morfologiche
    cv::Mat kernel_3;
    cv::Mat kernel_5;
    cv::Mat kernel_7;

    // Detection window
    cv::Rect detectionWindow;
    bool useWindow;

private:
    std::vector<ClusterData> cluster_points;

private:
    // Modifica la funzione helper per includere le informazioni di gruppo
    NDArray<float, 2> convertToNDArray(const Mat& normalizedData) {
        NDArray<float, 2> arrayData({(size_t)normalizedData.rows, (size_t)2});
        
        for(int i = 0; i < normalizedData.rows; i++) {
            arrayData[i][0] = normalizedData.at<float>(i, 0);
            arrayData[i][1] = normalizedData.at<float>(i, 1);
        }
        
        return arrayData;
    }

public:
    OpticalFlowTracker(int x = 0, int y = 0, int width = 0, int height = 0)
        : maxCorners(1000),
          qualityLevel(0.1), // previous 0.3
          minDistance(7.0), // previous 7.0
          blockSize(7),
          criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03),
          winSize(15, 15), // 21 21 
          maxLevel(2), // 3

          // Inizializzazione dei nuovi parametri
          magnitudeTolerance(2.5f),
          angleTolerance(0.35f),
          minOccurrences(7),

          frame_counter(0),
          detectionWindow(x, y, width, height),
          useWindow(width > 0 && height > 0)
    {
        backSub = cv::createBackgroundSubtractorMOG2(1, 200, false);
        kernel_3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
        kernel_5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
        kernel_7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));

        cv::RNG rng;
        for(int i = 0; i < 1000; i++) {
            colors.push_back(cv::Scalar(
                rng.uniform(0,255),
                rng.uniform(0,255),
                rng.uniform(0,255)
            ));
        }
    }

    void cleanup() {
        fgMask.release();
        line_mask.release();
        line_mask_kmeans.release();
        frame_gray.release();
        blurred_f.release();
        ROI_f.release();
        morph.release();
    }

    ~OpticalFlowTracker() {
        cleanup();
        if (backSub) {
            backSub.release();
        }
    }

    // Funzione per verificare se due movimenti sono simili
    bool areSimilarMovements(float mag1, float ang1, float mag2, float ang2, 
                            float magTolerance = 1.5f, float angTolerance = 0.15f) {
        // Verifica se il modulo è simile (entro la tolleranza)
        bool similarMagnitude = std::abs(mag1 - mag2) <= magTolerance;
        
        // Calcola la differenza angolare minima (considerando la circolarità)
        float angDiff = std::abs(ang1 - ang2);
        float twoPi = 2.0f * static_cast<float>(CV_PI);
        angDiff = std::min(angDiff, twoPi - angDiff);
        bool similarAngle = angDiff <= angTolerance;
        
        return similarMagnitude && similarAngle;
    }

    // Funzione per raggruppare i punti simili e contare le occorrenze
    std::vector<FeaturePoint> filterByOccurrences(const std::vector<Point2f>& current,
                                                const std::vector<Point2f>& previous) {
        std::vector<FeaturePoint> allFeatures;
        
        // Prima passiamo crea tutti i FeaturePoint
        for(size_t i = 0; i < current.size(); i++) {
            FeaturePoint fp;
            fp.position = current[i];
            fp.oldPosition = previous[i];
            
            allFeatures.push_back(fp);
        }
        
        return allFeatures;
    }

    std::vector<FeaturePoint> prepareFeaturePoints(
        const std::vector<Point2f>& current, const std::vector<Point2f>& previous) {
    
        // Prima filtra i punti basandosi sulle occorrenze
        return filterByOccurrences(current, previous);
    }

    Mat prepareDataForClustering(const vector<FeaturePoint>& features) {
        if(features.empty()) return Mat();

        // Aumentiamo il numero di colonne per includere le nuove feature
        Mat data(features.size(), 2, CV_32F);  // x,y

        // Trova i valori min e max per normalizzazione
        float minX = features[0].position.x, maxX = features[0].position.x;
        float minY = features[0].position.y, maxY = features[0].position.y;
        
        // Prima passata per trovare i range
        for(const auto& fp : features) {
            minX = min(minX, fp.position.x);
            maxX = max(maxX, fp.position.x);
            minY = min(minY, fp.position.y);
            maxY = max(maxY, fp.position.y);
        }
        
        // Seconda passata per riempire la matrice
        for(size_t i = 0; i < features.size(); i++) {
            data.at<float>(i, 0) = (features[i].position.x - minX) / (maxX - minX);
            data.at<float>(i, 1) = (features[i].position.y - minY) / (maxY - minY);
        }

        return data;
    }

    void init(cv::Mat& frame, int i) {
        if(frame.empty()) return;

        if (i == 0) {
            fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
            line_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
            line_mask_kmeans = cv::Mat::zeros(frame.size(), CV_8UC3);
        }

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, blurred_f, cv::Size(5, 5), 0);
        backSub->apply(blurred_f, fgMask, -1);
    }

    void initialize() {
        morph = fgMask.clone();
        cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel_5);
        cv::morphologyEx(morph, morph, cv::MORPH_OPEN, kernel_3);
        cv::dilate(morph, morph, kernel_3);

        cv::bitwise_and(frame_gray, morph, ROI_f);
        cv::goodFeaturesToTrack(ROI_f, shi_tomasi_keypoints,
            maxCorners,
            qualityLevel,
            minDistance,
            cv::Mat(),
            blockSize);

        old_gray = frame_gray.clone(); 
        frame_counter = 0;
        morph = fgMask.clone();
    }

    Mat process3D(Mat& img, const Mat& homography, const Mat& homography_inv){
        try{
            for (int i = 0; i < cluster_points.size(); i++){
                int npoints = cluster_points[i].points.size();
                vector<Point2f> kp_float = cluster_points[i].points;
                Scalar cluster_color = cluster_points[i].color;

                vector<Point2i> kp;
                kp.reserve(kp_float.size());  // Pre-alloca spazio per efficienza

                // Converte ogni punto da float a int
                for(const auto &pt : kp_float) {
                    kp.push_back(Point2i(cvRound(pt.x), cvRound(pt.y)));
                }

                //building cluster fake 2D bounding box
                vector<Point2i> h_ordered = kp;
                sort(h_ordered.begin(), h_ordered.end(),
                    [](const Point2i& a, const Point2i& b) {
                        return a.x < b.x;
                    });
                Point2i leftest = h_ordered[0];

                vector<Point2i> v_ordered = kp;
                sort(v_ordered.begin(), v_ordered.end(),
                    [](const Point2i& a, const Point2i& b) {
                        return a.y < b.y;
                    });
                Point2i highest = v_ordered[0];

                int twoDBB_w = abs(h_ordered[0].x - h_ordered[npoints-1].x);
                int twoDBB_h = abs(v_ordered[0].y - v_ordered[npoints-1].y);

                Mat twoDBB = (Mat_<float>(4,2) << 
                    leftest.x, highest.y,
                    leftest.x + twoDBB_w, highest.y,
                    leftest.x + twoDBB_w, highest.y + twoDBB_h,
                    leftest.x, highest.y + twoDBB_h);

                //euclidean coordinates -> homogenous coordinates: add 1 as the third element of the array
                Mat point = (Mat_<float>(3,1) << twoDBB.at<float>(3,0), twoDBB.at<float>(3,1), 1); // Mat point = Mat(3, 1, CV_32F, p);
                Mat result = homography_inv * point;

                //homogenous coordinates -> euclidean coordinates: DIVIDE the first two elements of the vector by the third
                Point2f leftMedian((float)(result.at<float>(0,0) / result.at<float>(2,0)), (float)(result.at<float>(1,0) / result.at<float>(2,0)));

                if (i == cluster_points.size() - 1) {
                    cout << "Stampa point " << endl;
                    for(size_t i = 0; i < point.rows; i++) {
                        for(size_t j = 0; j < point.cols; j++) {
                            cout << "punto x" << i << "punto y" << j << " " << point.at<float>(i, j) << endl;
                        }
                    }
                    cout << "Stampa leftMedian " << endl;
                    cout << "punto x" << leftMedian.x << "punto y" << leftMedian.y << endl;
                }

                point = (Mat_<float>(3,1) << twoDBB.at<float>(2,0), twoDBB.at<float>(2,1), 1);
                result = homography_inv * point;
                Point2f rightMedian((float)(result.at<float>(0,0) / result.at<float>(2,0)), (float)(result.at<float>(1,0) / result.at<float>(2,0)));

                //use the absolute difference between left and right to compute the size of the projected square in CHESSBOARD mesurements
                float deltax = std::abs(leftMedian.x - rightMedian.x);
                deltax = deltax/2;

                //homographedSquare is the base of the 3D bounding box expressed in homogenous chessboard coordinates
                Mat homographedSquare = (Mat_<float>(4, 3) <<
                    leftMedian.x,  leftMedian.y + deltax,  1.0f,
                    rightMedian.x, rightMedian.y + deltax, 1.0f,
                    rightMedian.x, rightMedian.y - deltax, 1.0f,
                    leftMedian.x,  leftMedian.y - deltax,  1.0f);

                //convert each point to pixel coordinates using homography matrix
                Mat hcImgSquare = Mat::zeros(4, 3, CV_32F);
                
                for (int j = 0; j < 4; j++) {
                    Mat point = (Mat_<float>(3,1) <<
                        homographedSquare.at<float>(j,0),
                        homographedSquare.at<float>(j,1),
                        homographedSquare.at<float>(j,2));
                    Mat result = homography * point;
                    //convert homogenous pixel coordinates in euclidean pixel coordinates
                    hcImgSquare.at<float>(j,0) = result.at<float>(0) / result.at<float>(2);
                    hcImgSquare.at<float>(j,1) = result.at<float>(1) / result.at<float>(2);
                    hcImgSquare.at<float>(j,2) = 0;
                }

                //imgSquare is the integer euclidean pixel coordinates of the base of the 3d bounding box
                Mat imgSquare;
                Mat subset = hcImgSquare.colRange(0, 2); //hcImgSquare[:,0:2].reshape((4,2)))
                subset.forEach<int>([](int& value, const int* position) -> void {
                    value = cvRound(value); // Arrotonda il valore
                }); 
                imgSquare = subset.clone();

                cout << "Stampa imgSquare: " << endl;
                cout << "imgSquare row size: " << imgSquare.rows << " imgSquare col size: " << imgSquare.cols << endl;
                for (int i = 0; i < imgSquare.rows; i++) {
                    for (int j = 0; j < imgSquare.cols; j++) {
                        cout << "punto: " << imgSquare.at<float>(i, j) << endl;
                    }
                }

                //draw3D(img, corners, imgpts);
                render3DBoundingBox(img, twoDBB, imgSquare, twoDBB_h, cluster_color);
                twoDBB.release();
                kp.clear();
            }
        } catch (const cv::Exception& e){
            cout << "error in process3D" << endl;
        }
        return img;
    }

    // Genera N colori diversi basati sul numero di cluster trovati
    std::vector<cv::Scalar> generateDistinctColors(int n) {
        std::vector<cv::Scalar> colors;
        
        // Colori base altamente contrastanti
        std::vector<cv::Scalar> baseColors = {
            cv::Scalar(0, 0, 255),     // Rosso
            cv::Scalar(0, 255, 255),   // Giallo
            cv::Scalar(255, 0, 0),     // Blu
            cv::Scalar(0, 255, 0),     // Verde
            cv::Scalar(255, 0, 255),   // Magenta
            cv::Scalar(255, 128, 0),   // Blu chiaro
            cv::Scalar(0, 128, 255),   // Arancione
            cv::Scalar(255, 0, 128)    // Viola
        };
        
        // Prendiamo i primi n colori necessari
        for(int i = 0; i < n && i < baseColors.size(); i++) {
            colors.push_back(baseColors[i]);
        }
        
        // Se servono più colori, li generiamo con HSV
        if(n > baseColors.size()) {
            float hueStep = 360.0f / (n - baseColors.size());
            for(int i = baseColors.size(); i < n; i++) {
                cv::Mat hsv(1, 1, CV_32FC3);
                hsv.at<cv::Vec3f>(0,0)[0] = hueStep * (i - baseColors.size()) / 2; // hue (0-180)
                hsv.at<cv::Vec3f>(0,0)[1] = 1.0f;  // saturazione massima
                hsv.at<cv::Vec3f>(0,0)[2] = 1.0f;  // valore massimo
                
                cv::Mat bgr;
                cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                cv::Vec3f color = bgr.at<cv::Vec3f>(0,0);
                colors.push_back(cv::Scalar(color[0] * 255, color[1] * 255, color[2] * 255));
            }
        }
        
        return colors;
    }

    std::tuple<cv::Mat, cv::Mat, cv::Mat> process(cv::Mat& frame) {
        if(frame.empty()) return std::make_tuple(cv::Mat(), cv::Mat(), cv::Mat());

        old_gray = frame_gray.clone();
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, blurred_f, cv::Size(5, 5), 0);

        if(frame_counter == 10) {
            backSub = cv::createBackgroundSubtractorMOG2(1, 200, false);
        }

        if(frame_counter >= 10) {
            backSub->apply(blurred_f, fgMask, -1);
        }

        if(frame_counter == 14) {
            line_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
            line_mask_kmeans = cv::Mat::zeros(frame.size(), CV_8UC3);
            old_keypoints = shi_tomasi_keypoints;

            morph = fgMask.clone();
            cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel_3);
            cv::morphologyEx(morph, morph, cv::MORPH_OPEN, kernel_3);
            cv::dilate(morph, morph, kernel_3);

            cv::bitwise_and(frame_gray, morph, ROI_f);
            cv::goodFeaturesToTrack(ROI_f, shi_tomasi_keypoints,
                maxCorners,
                qualityLevel,
                minDistance,
                cv::Mat(),
                blockSize);
        }

        std::vector<cv::Point2f> displaced_kp;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::Mat kmeansOutput;

        try {
            cv::calcOpticalFlowPyrLK(old_gray, frame_gray,
                shi_tomasi_keypoints, displaced_kp,
                status, err,
                winSize,
                maxLevel,
                criteria);
        } catch (const cv::Exception& e) {
            shi_tomasi_keypoints.clear();
            frame_counter = 0;
            return std::make_tuple(cv::Mat(), frame, fgMask);
        }

        std::vector<cv::Point2f> good_new, good_old;
        good_new.reserve(displaced_kp.size());
        good_old.reserve(displaced_kp.size());
        for(size_t i = 0; i < displaced_kp.size(); i++) {
            if(status[i]) {
                good_new.push_back(displaced_kp[i]);
                good_old.push_back(shi_tomasi_keypoints[i]);
            }
        }

        if(!good_new.empty()) {
            auto features = prepareFeaturePoints(good_new, good_old);
                
            std::vector<int> clusters;
                
            if(!features.empty()) {
                Mat data = prepareDataForClustering(features);
                if(!data.empty()) {
                    try {
                        // Raccogli i dati di foreground mantenendo la normalizzazione
                        std::vector<int> fgIndices;
                        for(int i = 0; i < data.rows; i++) {
                            fgIndices.push_back(i);
                        }

                        // Converti i punti in NDArray, includendo le informazioni di gruppo
                        NDArray<float, 2> pointsArray = convertToNDArray(data); //convertToNDArray(fgData);
                        
                        // Crea e applica DBSCAN con i parametri aggiustati per considerare le nuove feature
                        // Aumentiamo leggermente eps perché ora abbiamo più dimensioni
                        DBSCAN<float> dbscan(pointsArray, 0.2f, 6); // 0.3 8
                        dbscan.run();

                        // Ottieni le etichette
                        const auto& dbscanLabels = dbscan.labels();

                        int numClusters = dbscan.nClusters();

                        // DEBUG
                        /* std::cout << "DBSCAN ha creato " << numClusters << " cluster" << std::endl;
                        // Conta quanti punti per ogni cluster
                        std::vector<int> clusterCounts(numClusters + 1, 0); // +1 per i punti NOISY
                        for(size_t i = 0; i < dbscanLabels.size(); i++) {
                            if(dbscanLabels[i] == DBSCAN<float>::NOISY) {
                                clusterCounts[numClusters]++;  // L'ultimo indice per i punti NOISY
                            } else {
                                clusterCounts[dbscanLabels[i]]++;
                            }
                        }
                        for(int i = 0; i < numClusters; i++) {
                            std::cout << "Cluster " << i << ": " << clusterCounts[i] << " punti" << std::endl;
                        }
                        std::cout << "Punti NOISY: " << clusterCounts[numClusters] << std::endl; */

                        // Crea colori casuali per ogni cluster
                        std::vector<cv::Scalar> clusterColors = generateDistinctColors(numClusters); 

                        cluster_points.clear();
                        cluster_points.resize(numClusters + 1);
                        
                        for(size_t i = 0; i < dbscanLabels.size(); i++) {
                            int originalIndex = fgIndices[i];  // Indice nel vettore features originale
                            cv::Scalar color;
                            cv::Scalar white;

                            if (i == dbscanLabels.size() - 1) {
                                white = Scalar(0, 255, 0);
                                cluster_points[numClusters].points.push_back(Point2f(1100.0, 700.0));
                                cluster_points[numClusters].points.push_back(Point2f(900.0, 600.0));
                                cluster_points[numClusters].points.push_back(Point2f(950.0, 750.0));
                                cluster_points[numClusters].color = white;

                                circle(frame,
                                    cv::Point(1100, 700),
                                    5, white, -1);
                                circle(frame,
                                    cv::Point(900, 600),
                                    5, white, -1);
                                circle(frame,
                                    cv::Point(950, 750),
                                    5, white, -1);
                            } else {
                                if(dbscanLabels[i] == DBSCAN<float>::NOISY) {
                                    color = cv::Scalar(0, 0, 0);
                                } else {
                                    color = clusterColors[dbscanLabels[i]];
                                    cluster_points[dbscanLabels[i]].points.push_back(features[originalIndex].position);
                                    cluster_points[dbscanLabels[i]].color = color;
                                }
                                circle(frame,
                                    cv::Point(features[originalIndex].position.x, features[originalIndex].position.y),
                                    5, color, -1);
                                line(line_mask,
                                    cv::Point(features[originalIndex].position.x, features[originalIndex].position.y),
                                    cv::Point(features[originalIndex].oldPosition.x, features[originalIndex].oldPosition.y),
                                    color, 2);
                            }  
                        }
                    } catch (const cv::Exception& e) {
                        cout << "Errore cluster!! " << endl;
                        clusters.resize(features.size(), 0);
                    }
                } else {
                    clusters.resize(features.size(), 0);
                }
            }
        }

        cv::Mat output;
        cv::add(frame, line_mask, output);

        shi_tomasi_keypoints = displaced_kp;
        frame_counter = (frame_counter + 1) % 15;

        return std::make_tuple(kmeansOutput, output, morph);
    }

    Mat render3DBoundingBox(Mat& img, const Mat& clusterBB, const Mat& cubicBBBase, int delta_Z, const Scalar& color) {
        cout << "dentro3dbb" << endl;
        // Disegna il bounding box 2D in rosso
        //drawRectangle(img, clusterBB, Scalar(0, 0, 255));

        cout << "stampa cubicbbbase" << endl;
        for (size_t i = 0; i < cubicBBBase.rows; i++) {
            for (size_t j = 0; j < cubicBBBase.rows; j++) {
                cout << "punto(x,y) " << cubicBBBase.at<int>(i, j) << endl;
            }
        }
        cout << "color post cubicccccc " << color << endl;

        // Disegna la base del bounding box 3D
        drawRectangle(img, cubicBBBase, color);
        Mat cubicBBTip = cubicBBBase.clone();

        // Modifica le coordinate y sottraendo delta_Z
        for(int i = 0; i < cubicBBTip.rows; i++) {
            cubicBBTip.at<float>(i, 1) = cubicBBBase.at<float>(i, 1) - delta_Z;
        }
        cout << "dopo for dentro3dbb" << endl;
        // Disegna la punta del bounding box 3D
        drawRectangle(img, cubicBBTip, color);
        cout << "dopo 2ndo draw" << endl;
        // Connette la base con la punta
        for(int i = 0; i < 4; i++) {
            Point2f base_point(cubicBBBase.at<float>(i, 0), cubicBBBase.at<float>(i, 1));
            Point2f tip_point(cubicBBTip.at<float>(i, 0), cubicBBTip.at<float>(i, 1));
            line(img, base_point, tip_point, color, 2);
        }
        cout << "fine draw" << endl;
        return img;
    }

    Mat drawRectangle(Mat& img, const Mat& rectangle, const Scalar& color) {
        for(int i = 0; i < 4; i++) {
            Point2f current_point(rectangle.at<float>(i, 0), rectangle.at<float>(i, 1));
            Point2f next_point;

            if(i == 3) {
                next_point = Point2f(rectangle.at<float>(0, 0), rectangle.at<float>(0, 1));
            } else {
                next_point = Point2f(rectangle.at<float>(i + 1, 0), rectangle.at<float>(i + 1, 1));
            }

            line(img, current_point, next_point, color, 2);
        }

        return img;
    }
};

string generateRandomId(int length) {
    const string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<> distribution(0, chars.size() - 1);

    string randomId;
    for (int i = 0; i < length; ++i) {
        randomId += chars[distribution(generator)];
    }
    return randomId;
}

class VideoServer {
    string cameraId;
    int windowX, windowY, windowWidth, windowHeight;
    bool useWindow;
    bool showDisplays;
public:
    VideoServer(int camera, string file, string id = "",
                int x = 0, int y = 0, int width = 0, int height = 0,
                bool displays = true)
        : running(false), windowX(x), windowY(y),
          windowWidth(width), windowHeight(height),
          showDisplays(displays) {

        useWindow = (width > 0 && height > 0);
        cameraId = id.empty() ? generateRandomId(10) : id;

        server.init_asio();
        server.set_access_channels(websocketpp::log::alevel::none);
        server.clear_access_channels(websocketpp::log::alevel::all);

        server.set_validate_handler([this](connection_hdl hdl) -> bool {
            auto con = server.get_con_from_hdl(hdl);
            string resource = con->get_resource();
            return resource == "/camera" + this->cameraId;
        });

        server.set_open_handler(bind(&VideoServer::on_open, this, placeholders::_1));
        server.set_close_handler(bind(&VideoServer::on_close, this, placeholders::_1));

        if (showDisplays) {
            cv::namedWindow("colours", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("bw", cv::WINDOW_AUTOSIZE);
        }

        videoThread = thread(&VideoServer::processVideo, this, camera, file);
    }

    ~VideoServer() {
        if (showDisplays) {
            cv::destroyWindow("colours");
            cv::destroyWindow("bw");
        }
    }

    string getCameraId() const { return cameraId; }

    void run(uint16_t port) {
        server.listen(port);
        server.start_accept();
        try {
            server.run();
        } catch (const exception& e) {
            cerr << "Server error: " << e.what() << endl;
        }
    }

    void stop() {
        running = false;
        if (videoThread.joinable()) {
            videoThread.join();
        }
        server.stop();
    }

private:
    void on_open(connection_hdl hdl) {
        lock_guard<mutex> lock(connectionsMutex);
        connections.insert(hdl);
        cout << "Client connected. Total clients: " << connections.size() << endl;
    }

    void on_close(connection_hdl hdl) {
        lock_guard<mutex> lock(connectionsMutex);
        connections.erase(hdl);
        cout << "Client disconnected. Total clients: " << connections.size() << endl;
    }

    void processVideo(int camera, string file) {
        VideoCapture cap;
        if (file.empty())
            cap.open(camera);
        else {
            file = samples::findFileOrKeep(file);
            cap.open(file);
        }

        if (!cap.isOpened()) {
            cerr << "Cannot open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
            exit(1);
        }

        double fps = cap.get(CAP_PROP_FPS);
        int delay = 1000 / (fps > 0 ? fps : 30);

        OpticalFlowTracker tracker(windowX, windowY, windowWidth, windowHeight);
        Mat frame;
        running = true;

        // Training phase - first 10 frames
        for(int i = 0; i < 10; i++) {
            cap >> frame;
            if(frame.empty()) break;
            tracker.init(frame, i);
        }

        while (running) {
            cap >> frame;
            if (frame.empty()) {
                if (!file.empty()) {
                    cap.set(CAP_PROP_POS_FRAMES, 0);
                    continue;
                }
                break;
            }

            int64 t = getTickCount();
            auto [kmeans, processed, morph] = tracker.process(frame);
            t = getTickCount() - t;

            double fps = getTickFrequency() / (double)t;

            if (showDisplays) {
                cv::imshow("colours", processed);
                cv::imshow("bw", morph);

                char key = (char)cv::waitKey(delay);
                if (key == 27) {  // ESC
                    running = false;
                    break;
                }
            }

            // Gestione WebSocket
            Mat resized;
            resize(processed, resized, Size(), 0.5, 0.5);
            vector<uchar> buffer;
            vector<int> params = {IMWRITE_JPEG_QUALITY, 60};
            imencode(".jpg", resized, buffer, params);

            ostringstream jsonStream;
            jsonStream << "{";
            jsonStream << "\"fps\":" << fixed << setprecision(1) << fps;
            jsonStream << "}";
            string jsonMessage = jsonStream.str();

            lock_guard<mutex> lock(connectionsMutex);
            for (auto& hdl : connections) {
                try {
                    server.send(hdl, jsonMessage, websocketpp::frame::opcode::text);
                    server.send(hdl, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary);
                } catch (const websocketpp::exception& e) {
                    cerr << "Send error: " << e.what() << endl;
                }
            }

            this_thread::sleep_for(chrono::milliseconds(delay));
        }

        cap.release();
    }

    Server server;
    set<connection_hdl, owner_less<connection_hdl>> connections;
    mutex connectionsMutex;
    thread videoThread;
    atomic<bool> running;
};


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

static bool loadCameraParams(const string& filename, Size& imageSize, Size& boardSize, Mat& cameraMatrix, Mat& distCoeffs, float& squareSize, double& totalAvgErr)
{
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

pair<Mat,Mat> calc_homography(const string &filePath, Mat& cameraMatrix, Mat& distCoeffs, Size& boardSize)
{
    Mat H_float, H_inv;
    try{
        //termination criteria
        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);

        //! [compute-image-points]
        vector<Point3f> objectPoints;
        vector<Point2f> imagePoints;
        Mat img = imread( samples::findFile( filePath) );
        Size imageSize = img.size();
        Mat newcameramtx = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);

        Mat undistorted; // Crea una nuova Mat per l'output
        undistort(img, undistorted, cameraMatrix, distCoeffs, newcameramtx);
        Mat gray;
        cvtColor(undistorted, gray, COLOR_BGR2GRAY);

       //prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        Mat objp = Mat::zeros(boardSize.width * boardSize.height, 3, CV_32F);
        for(int i = 0; i < boardSize.height; i++) {
            for(int j = 0; j < boardSize.width; j++) {
                objp.at<float>(i * boardSize.width + j, 0) = j;
                objp.at<float>(i * boardSize.width + j, 1) = i;
                // La terza coordinata rimane 0
            }
        }

        vector<Point3f> axis = {Point3f(0, 0, 0), Point3f(0, 3, 0), Point3f(3, 3, 0), Point3f(3, 0, 0),
                                Point3f(0, 0, -3), Point3f(0, 3, -3), Point3f(3, 3, -3), Point3f(3, 0, -3)};

        // Trova gli angoli della scacchiera
        vector<Point2f> corners;
        bool found = findChessboardCorners(undistorted, boardSize, corners);
        if (!found) {
            throw runtime_error("Impossibile trovare gli angoli della scacchiera");
        }

        cornerSubPix(gray, corners, Size(11,11), Size(-1,-1), criteria);
        //Find the rotation and translation vectors.
        Mat rvecs, tvecs;
        solvePnP(objp, corners, newcameramtx, distCoeffs, rvecs, tvecs);
        // project 3D points to image plane (USED ONLY FOR OPENCV DEMO CUBE)
        vector<Point2f> imgpts_float;
        projectPoints(axis, rvecs, tvecs, newcameramtx, distCoeffs, imgpts_float);
        vector<Point2i> imgpts;

        // Converte ogni punto da float a int
        for(const auto &pt : imgpts_float) {
            imgpts.push_back(Point2i(cvRound(pt.x), cvRound(pt.y)));
        }

        //computing homography matrix and it inverse
        Mat H = findHomography(objp, corners);
        if(H.type() != CV_32F) {
            H.convertTo(H_float, CV_32F);
        } else {
            H_float = H;
        }
        H_inv = H_float.inv();
    } catch(const cv::Exception& e){
        cout << "Errore Homography" << endl;
    }
    return {H_float, H_inv};
}


int main(int argc, char** argv) {

    float squareSize, grid_width, aspectRatio = 1;
    int flags = 0;
    double totalAvgErr = 0;
    Mat cameraMatrix, distCoeffs;
    string outputFilename;
    string images_path;
    Size imageSize, boardSize;
    Pattern pattern = CHESSBOARD;
    vector<vector<cv::Point3f>> objectPoints; // Punti 3D del mondo reale
    vector<vector<cv::Point2f>> imagePoints; // Punti 2D nelle immagini

    CommandLineParser parser(argc, argv,
        "{ help    |   | print help message }"
        "{ camera c | 0 | capture video from camera (device index starting from 0) }"
        "{ video v  |   | use video as input }"
        "{ port p   | 5555 | WebSocket server port }"
        "{ id      |   | camera identifier for the stream endpoint }"
        "{ x       | 0 | x coordinate of detection window }"
        "{ y       | 0 | y coordinate of detection window }"
        "{ width w | 0 | width of detection window (0 for full frame) }"
        "{ height h| 0 | height of detection window (0 for full frame) }"
        "{ display d| 0 | show display windows (0 or 1) }"
        "{ images_path  | res/calibration | path to chessboard images }"
        "{ output_file  | out_camera_data.yml | output filename for calibration parameters }"
        "{ board_width  | 9 | chessboard width }"
        "{ board_height | 6 | chessboard height }"
        "{ square_size  | 0.025 | chessboard square size }");

    parser.about("Optical Flow Tracking with WebSocket streaming and display capability");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    int camera = parser.get<int>("camera");
    string file = parser.get<string>("video");
    int port = parser.get<int>("port");
    string cameraId = parser.get<string>("id");
    bool showDisplays = parser.get<int>("display") != 0;

    int x = parser.get<int>("x");
    int y = parser.get<int>("y");
    int width = parser.get<int>("w");
    int height = parser.get<int>("h");

    boardSize.width = parser.get<int>("board_width");
    boardSize.height = parser.get<int>("board_height");
    squareSize = (float) parser.get<double>("square_size");
    images_path = parser.get<String>("images_path");
    outputFilename = parser.get<String>("output_file");
    grid_width = squareSize * (boardSize.width - 1);
    cameraMatrix = Mat::eye(3, 3, CV_64F);

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    //start CALIBRATION AND PARAMS SAVED
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

            //drawChessboardCorners(img_corners, boardSize, corners, found);
            //imshow("Chessboard corners detection", img_corners);
            //waitKey();
        }

        if( imagePoints.size() > 0 )
            runAndSave(outputFilename, imagePoints, imageSize,
                                   boardSize, pattern, squareSize, grid_width,
                                   aspectRatio, flags, cameraMatrix, distCoeffs);
    }
    //end CALIBRATION AND PARAMS SAVED

    //omografia
    cout << "Prima di Homography" << endl;
    auto [homography, homographyInv] = calc_homography("res/floor_surface/piano_pav (4).jpg", cameraMatrix, distCoeffs, boardSize);
    cout << "<Dopo> Homography" << endl;

    VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else {
        file = samples::findFileOrKeep(file);
        cap.open(file);
    }

    if (!cap.isOpened()) {
        cerr << "Cannot open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int delay = 1000 / (fps > 0 ? fps : 30);

    OpticalFlowTracker tracker(x, y, width, height);
    Mat frame;
    
    //namedWindow("kmeans", WINDOW_AUTOSIZE);
    namedWindow("dbscan", WINDOW_AUTOSIZE);
    //namedWindow("bw-morph", WINDOW_AUTOSIZE);

    // Training phase - first 10 frames
    for(int i = 0; i < 10; i++) {
        cap >> frame;
        if(frame.empty()) break;
        tracker.init(frame, i);
    }

    tracker.initialize();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            if (!file.empty()) {
                cap.set(CAP_PROP_POS_FRAMES, 0);
                continue;
            }
            break;
        }

        int64 t = getTickCount();
        auto [kmeans, processed, morph] = tracker.process(frame);
        t = getTickCount() - t;
        double fps = getTickFrequency() / (double)t;

        putText(processed, "FPS: " + to_string(int(fps)), Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 2);

        Mat res_img = tracker.process3D(processed, homography, homographyInv);
        imshow("result", res_img);
        //imshow("kmeans", kmeans);
        imshow("dbscan", processed);
        //imshow("bw-morph", morph);

        char key = (char)waitKey(delay);
        if (key == 27) break;  // ESC
    }

    cap.release();
    destroyAllWindows();

    /* try {
        cout << "Video server started on port " << port << endl;
        cout << "Stream available at: /camera" << cameraId << endl;
        if (showDisplays) {
            cout << "Display windows enabled - press ESC to quit" << endl;
        }

        VideoServer server(camera, file, cameraId, x, y, width, height, showDisplays);
        server.run(port);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    } */

    return 0;
}