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
    // Struttura per raggruppare punti simili
    struct MovementGroup {
        float magnitude;
        float angle;
        int count;
        std::vector<size_t> pointIndices;  // Indici dei punti che appartengono a questo gruppo
    };

    struct FeaturePoint {
        Point2f position;      // posizione corrente
        Point2f oldPosition;   // posizione precedente
        float magnitude;       // modulo del vettore spostamento
        float angle;          // verso (angolo) del vettore spostamento
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
    cv::Mat mask;
    cv::Mat fgMask;
    cv::Mat old_gray;
    cv::Mat line_mask;
    std::vector<cv::Point2f> shi_tomasi_keypoints;
    std::vector<cv::Point2f> old_keypoints;
    std::vector<cv::Scalar> colors;
    int frame_counter;
    int lastForegroundCount;

    // Kernels per trasformazioni morfologiche
    cv::Mat kernel_3;
    cv::Mat kernel_5;
    cv::Mat kernel_7;

    // Detection window
    cv::Rect detectionWindow;
    bool useWindow;

private:
    // Modifica la funzione helper per includere le informazioni di gruppo
    NDArray<float, 2> convertToNDArray(const Mat& normalizedData) {
        NDArray<float, 2> arrayData({(size_t)normalizedData.rows, (size_t)normalizedData.cols});
        
        for(int i = 0; i < normalizedData.rows; i++) {
            for(int j = 0; j < normalizedData.cols; j++) {
                arrayData[i][j] = normalizedData.at<float>(i, j);
            }
        }
        
        return arrayData;
    }

public:
    OpticalFlowTracker(int x = 0, int y = 0, int width = 0, int height = 0)
        : maxCorners(1000),
          qualityLevel(0.3),
          minDistance(5.0),
          blockSize(7),
          criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03),
          winSize(15, 15),
          maxLevel(2),

          // Inizializzazione dei nuovi parametri
          magnitudeTolerance(2.5f),
          angleTolerance(0.35f),
          minOccurrences(7),

          frame_counter(0),
          lastForegroundCount(0),
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
    std::pair<std::vector<FeaturePoint>, std::vector<MovementGroup>> filterByOccurrences(const std::vector<Point2f>& current,
                                                const std::vector<Point2f>& previous,
                                                int minOccurrences,
                                                float magTolerance,
                                                float angTolerance) {
        std::vector<FeaturePoint> allFeatures;
        std::vector<MovementGroup> groups;
        
        // Prima passiamo crea tutti i FeaturePoint
        for(size_t i = 0; i < current.size(); i++) {
            FeaturePoint fp;
            fp.position = current[i];
            fp.oldPosition = previous[i];
            
            // Calcola il vettore spostamento
            Point2f diff = current[i] - previous[i];
            
            // Calcola modulo
            fp.magnitude = sqrt(diff.x*diff.x + diff.y*diff.y);
            
            // Calcola angolo (verso) in radianti
            fp.angle = atan2(diff.y, diff.x);
            if(fp.angle < 0) {
                fp.angle += 2.0f * static_cast<float>(CV_PI);
            }
            
            allFeatures.push_back(fp);
        }
        
        // Raggruppa i punti con movimento simile
        std::vector<bool> assigned(allFeatures.size(), false);
        for(size_t i = 0; i < allFeatures.size(); i++) {
            if(assigned[i]) continue;
            
            MovementGroup newGroup;
            newGroup.magnitude = allFeatures[i].magnitude;
            newGroup.angle = allFeatures[i].angle;
            newGroup.count = 1;
            newGroup.pointIndices.push_back(i);
            assigned[i] = true;
            
            // Cerca altri punti con movimento simile
            for(size_t j = i + 1; j < allFeatures.size(); j++) {
                if(!assigned[j] && areSimilarMovements(
                    newGroup.magnitude, newGroup.angle,
                    allFeatures[j].magnitude, allFeatures[j].angle,
                    magTolerance, angTolerance)) {
                    newGroup.count++;
                    newGroup.pointIndices.push_back(j);
                    assigned[j] = true;
                }
            }
            
            groups.push_back(newGroup);
        }
        
        return {allFeatures, groups};
    }

    std::pair<std::vector<FeaturePoint>, std::vector<MovementGroup>> prepareFeaturePoints(
        const std::vector<Point2f>& current,
        const std::vector<Point2f>& previous,
        int minOccurrences,
        float magTolerance,
        float angTolerance) {
    
        // Prima filtra i punti basandosi sulle occorrenze
        return filterByOccurrences(current, previous, minOccurrences, magTolerance, angTolerance);
    }

    Mat prepareDataForClustering(const vector<FeaturePoint>& features, const vector<MovementGroup>& groups) {
        if(features.empty()) return Mat();

        // Aumentiamo il numero di colonne per includere le feature dei gruppi
        Mat data(features.size(), 6, CV_32F);  // x,y,magnitude,angle + group_magnitude,group_size

        // Trova i valori min e max per normalizzazione
        float minX = features[0].position.x, maxX = features[0].position.x;
        float minY = features[0].position.y, maxY = features[0].position.y;
        float minMag = features[0].magnitude, maxMag = features[0].magnitude;
        float minGroupMag = groups[0].magnitude, maxGroupMag = groups[0].magnitude;
        float minGroupSize = groups[0].count, maxGroupSize = groups[0].count;
        
        // Prima passata per trovare i range
        for(const auto& fp : features) {
            minX = min(minX, fp.position.x);
            maxX = max(maxX, fp.position.x);
            minY = min(minY, fp.position.y);
            maxY = max(maxY, fp.position.y);
            minMag = min(minMag, fp.magnitude);
            maxMag = max(maxMag, fp.magnitude);
        }
        
        for(const auto& group : groups) {
            minGroupMag = min(minGroupMag, group.magnitude);
            maxGroupMag = max(maxGroupMag, group.magnitude);
            minGroupSize = min(minGroupSize, (float)group.count);
            maxGroupSize = max(maxGroupSize, (float)group.count);
        }

        // Seconda passata per riempire la matrice
        for(size_t i = 0; i < features.size(); i++) {
            // Feature normali normalizzate
            data.at<float>(i, 0) = (features[i].position.x - minX) / (maxX - minX + 1e-5);
            data.at<float>(i, 1) = (features[i].position.y - minY) / (maxY - minY + 1e-5);
            data.at<float>(i, 2) = (features[i].magnitude - minMag) / (maxMag - minMag + 1e-5);
            data.at<float>(i, 3) = features[i].angle / (2 * CV_PI);

            // Trova il gruppo corrispondente e aggiungi le sue feature
            for(const auto& group : groups) {
                if(std::find(group.pointIndices.begin(), group.pointIndices.end(), i) != group.pointIndices.end()) {
                    data.at<float>(i, 4) = (group.magnitude - minGroupMag) / (maxGroupMag - minGroupMag + 1e-5);
                    data.at<float>(i, 5) = (group.count - minGroupSize) / (maxGroupSize - minGroupSize + 1e-5);
                    break;
                }
            }
        }

        return data;
    }

    void init(cv::Mat& frame, int i) {
        if(frame.empty()) return;

        if (i == 0) {
            fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
            mask = cv::Mat::zeros(frame.size(), CV_8UC3);
            line_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
        }

        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::Mat blurred_f;
        cv::GaussianBlur(frame_gray, blurred_f, cv::Size(5, 5), 0);
        backSub->apply(blurred_f, fgMask, -1);

        old_gray = frame_gray.clone();
    }

    // Genera N colori diversi basati sul numero di cluster trovati
    std::vector<cv::Scalar> generateDistinctColors(int n) {
        std::vector<cv::Scalar> colors(n);
        
        for(int i = 0; i < n; i++) {
            float hue = 360.0f * i / n;
            float sat = 0.9f;    // Saturazione fissa alta
            float val = 0.9f;    // Valore fisso alto per garantire colori brillanti
            
            cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(hue, sat, val));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            
            cv::Vec3f color = bgr.at<cv::Vec3f>(0, 0);
            colors[i] = cv::Scalar(
                color[0] * 255,
                color[1] * 255,
                color[2] * 255
            );
        }
        return colors;
    }

    std::pair<cv::Mat, cv::Mat> process(cv::Mat& frame) {
        if(frame.empty()) return {frame, cv::Mat()};

        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::Mat blurred_f;
        cv::GaussianBlur(frame_gray, blurred_f, cv::Size(5, 5), 0);

        if(frame_counter == 10) {
            backSub = cv::createBackgroundSubtractorMOG2(1, 200, false);
        }

        if(frame_counter >= 10) {
            backSub->apply(blurred_f, fgMask, -1);
        }

        if(frame_counter == 15) {
            line_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
            old_keypoints = shi_tomasi_keypoints;

            cv::Mat morph = fgMask.clone();
            cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel_3);
            cv::morphologyEx(morph, morph, cv::MORPH_OPEN, kernel_3);
            cv::dilate(morph, morph, kernel_3);

            cv::Mat ROI_f;
            cv::bitwise_and(frame_gray, morph, ROI_f);
            cv::goodFeaturesToTrack(ROI_f, shi_tomasi_keypoints,
                maxCorners,
                qualityLevel,
                minDistance,
                cv::Mat(),
                blockSize);

            if(shi_tomasi_keypoints.empty()) {
                cv::goodFeaturesToTrack(frame_gray, shi_tomasi_keypoints,
                    maxCorners,
                    qualityLevel,
                    minDistance,
                    cv::Mat(),
                    blockSize);
            }
        }

        if(frame_counter == 0 || shi_tomasi_keypoints.empty()) {
            cv::Mat morph = fgMask.clone();
            cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel_5);
            cv::morphologyEx(morph, morph, cv::MORPH_OPEN, kernel_3);
            cv::dilate(morph, morph, kernel_3);

            cv::Mat ROI_f;
            cv::bitwise_and(frame_gray, morph, ROI_f);
            cv::goodFeaturesToTrack(ROI_f, shi_tomasi_keypoints,
                maxCorners,
                qualityLevel,
                minDistance,
                cv::Mat(),
                blockSize);

            if(shi_tomasi_keypoints.empty()) {
                cv::goodFeaturesToTrack(frame_gray, shi_tomasi_keypoints,
                    maxCorners,
                    qualityLevel,
                    minDistance,
                    cv::Mat(),
                    blockSize);
            }

            old_keypoints = shi_tomasi_keypoints;
            old_gray = frame_gray.clone();
            frame_counter = (frame_counter + 1) % 30;
            return {frame, fgMask};
        }

        std::vector<cv::Point2f> displaced_kp;
        std::vector<uchar> status;
        std::vector<float> err;

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
            return {frame, fgMask};
        }

        std::vector<cv::Point2f> good_new, good_old;
        for(size_t i = 0; i < displaced_kp.size(); i++) {
            if(status[i]) {
                good_new.push_back(displaced_kp[i]);
                good_old.push_back(shi_tomasi_keypoints[i]);
            }
        }

        if(!good_new.empty()) {
            auto [features, groups] = prepareFeaturePoints(
                good_new, 
                good_old,
                minOccurrences,
                magnitudeTolerance, 
                angleTolerance
            );
                
            std::vector<int> clusters;
                
            // TODO: Check
            if(features.size() < 4) {
                clusters.resize(features.size(), 0);
            } else {
                Mat data = prepareDataForClustering(features, groups);
                if(!data.empty()) {
                    Mat labels;
                    Mat centers;

                    try {
                        kmeans(data, 2, labels,
                            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
                            3, KMEANS_PP_CENTERS, centers);

                        // DEBUG
                        /* int count0 = 0, count1 = 0;
                        for(int i = 0; i < labels.rows; i++) {
                            if(labels.at<int>(i) == 0) count0++;
                            else count1++;
                        }
                        std::cout << "K-means: cluster 0: " << count0 << " punti, cluster 1: " << count1 << " punti" << std::endl; */


                        // Raccogli i dati di foreground mantenendo la normalizzazione
                        Mat fgData;
                        std::vector<int> fgIndices;
                        for(int i = 0; i < labels.rows; i++) {
                            if(labels.at<int>(i) == 1) {
                                fgIndices.push_back(i);
                                fgData.push_back(data.row(i));
                            }
                        }

                        if(!fgData.empty()) {
                            try {
                                // Converti i punti in NDArray, includendo le informazioni di gruppo
                                NDArray<float, 2> pointsArray = convertToNDArray(fgData);
                                
                                // Crea e applica DBSCAN con i parametri aggiustati per considerare le nuove feature
                                // Aumentiamo leggermente eps perché ora abbiamo più dimensioni
                                DBSCAN<float> dbscan(pointsArray, 1.00f, 4, 4);
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

                                // Dopo DBSCAN
                                for(size_t i = 0; i < dbscanLabels.size(); i++) {
                                    int originalIndex = fgIndices[i];  // Indice nel vettore features originale
                                    cv::Scalar color;
                                    
                                    if(dbscanLabels[i] == DBSCAN<float>::NOISY) {
                                        color = cv::Scalar(0, 0, 0);
                                        circle(frame,
                                        cv::Point(features[originalIndex].position.x, features[originalIndex].position.y),
                                            5, color, -1);
                                        line(line_mask,
                                            cv::Point(features[originalIndex].position.x, features[originalIndex].position.y),
                                            cv::Point(features[originalIndex].oldPosition.x, features[originalIndex].oldPosition.y),
                                            color, 2);
                                    } else {
                                        color = clusterColors[dbscanLabels[i]];
                                        circle(frame,
                                            cv::Point(features[originalIndex].position.x, features[originalIndex].position.y),
                                            5, color, -1);
                                        line(line_mask,
                                            cv::Point(features[originalIndex].position.x, features[originalIndex].position.y),
                                            cv::Point(features[originalIndex].oldPosition.x, features[originalIndex].oldPosition.y),
                                            color, 2);
                                    }
                                }
                            } catch (const std::exception& e) {
                                std::cerr << "DBSCAN error: " << e.what() << std::endl;
                                clusters.resize(features.size(), 0);
                            }
                        }
                    } catch (const cv::Exception& e) {
                        clusters.resize(features.size(), 0);
                    }
                } else {
                    clusters.resize(features.size(), 0);
                }
            }
        }

        cv::Mat output;
        cv::add(frame, line_mask, output);

        old_gray = frame_gray.clone();
        shi_tomasi_keypoints = displaced_kp;
        frame_counter = (frame_counter + 1) % 30;

        cv::Mat morph = fgMask.clone();
        cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel_3);
        cv::morphologyEx(morph, morph, cv::MORPH_OPEN, kernel_3);
        cv::dilate(morph, morph, kernel_3);

        return {output, morph};
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
            auto [processed, morph] = tracker.process(frame);
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
    
    namedWindow("colours", WINDOW_AUTOSIZE);
    namedWindow("bw", WINDOW_AUTOSIZE);

    // Training phase - first 10 frames
    for(int i = 0; i < 10; i++) {
        cap >> frame;
        if(frame.empty()) break;
        tracker.init(frame, i);
    }

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
        auto [processed, morph] = tracker.process(frame);
        t = getTickCount() - t;
        double fps = getTickFrequency() / (double)t;

        putText(processed, "FPS: " + to_string(int(fps)), Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 2);

        imshow("colours", processed);
        imshow("bw", morph);

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