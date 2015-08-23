/** Calculo de un mapa de disparidad mediante el algoritmo Block matching **/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <opencvblobslib/BlobResult.h>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobOperators.h>
#include "SCalibData.h"

#define DEBUG 1

using namespace cv;
using namespace std;

//Variables globales. Esto solo es necesario para el modo debug, deberia quitarse mas adelante
Mat image[2], disp, disp8;

//Strings Trackbars y WinNames
String trackWindow = "Settings", disparityWindow = "Disparidad";
String sad_win_size_trackbar = "SAD Window Size";
String pre_filter_size_trackbar = "Pre-Filter Size";
String min_blob_area_trackbar = "Blob Area";

//Parametros Camara
SCalibData calibData;

//Parametros de ajuste BM
int sadWindowsize = 9, sadWindowsize_tmp = sadWindowsize;
int numberOfDisparities = 7;
int preFilterSize = 5, preFilterSize_tmp = preFilterSize;
int preFilterCap = 31;
int minDisparity = 0;
int textureThreshold = 10;
int uniqnessRatio = 15;
int speckleWindowSize = 100;
int speckleRange = 32;
int disp12MaxDiff = -1;
int thresholdRange = 0;
bool minDisparityNeg = false;

//Parametros ajuste Blobs
int min_blob_area = 80;

//Nombre Argumentos
String calib_filename_arg = "-c";
String distance_method_arg = "-d";

//Argumentos del programa
int camera1, camera2;
String calib_filename = "stereo_calib.yml";
int distance_method = 1;

void args(int argc, char **argv){
    camera1 = stoi(argv[1]); camera2 = stoi(argv[2]);
    for(int i=3; i < argc; i++){
        if(calib_filename_arg.compare(argv[i]) == 0){
            i++; calib_filename = argv[i];
        }else if(distance_method_arg.compare(argv[i]) == 0){
            i++; distance_method = stoi(argv[i]);
        }
    }
}

//Funciones de callback
void sad_window_size_callback(int v, void*){
    if(v % 2 == 0){
        if(v < sadWindowsize_tmp) sadWindowsize--;
        else sadWindowsize++;
        if(v == -1) sadWindowsize = 1;
        sadWindowsize_tmp = sadWindowsize;
        setTrackbarPos(sad_win_size_trackbar, trackWindow, sadWindowsize);
    }
}

void pre_filter_size_callback(int v, void*){
    if(v % 2 == 0){
        if(v <  preFilterSize_tmp) preFilterSize--;
        else preFilterSize++;
        if(v < 5) preFilterSize = 5;
        preFilterSize_tmp = preFilterSize;
        setTrackbarPos(pre_filter_size_trackbar, trackWindow, preFilterSize);
    }
}

double media(Mat roi);

double moda(Mat roi, Mat roi8);

//Variables del callback
String distance_window = "Distancia";
bool drawing = false;
int xi = -1, yi = -1;

static void disp_window_mouse_callback(int event, int x, int y, int flags, void* userdata){
    //if(event != EVENT_LBUTTONDOWN) return;
    /*cout << "x: " << x << " y: " << y << endl;
    Mat* depth_map = (Mat*) userdata;
    Vec3f vec = depth_map->at<Vec3f>(y,x);
    cout << "x: " << vec[0] << " y: " << vec[1] << " z: " << vec[2] << endl;*/
    if(event == EVENT_LBUTTONDOWN){
        drawing = true;
        xi = x;
        yi = y;
    }else if(event == EVENT_MOUSEMOVE && drawing){
        rectangle(disp8, Rect(Point(xi,yi), Point(x,y)), Scalar(0,255,0), 3);
        imshow(disparityWindow, disp8);
    }else if(event == EVENT_LBUTTONUP){
        drawing = false;
        if(xi == x && yi == y) return;
        //Matrices necesarias
        Rect rect(Point(xi,yi), Point(x,y));
        Mat roi(disp, rect), roi8(disp8, rect);
        //Calcular distancia
        double distance;
        if(distance_method == 1)
            distance = media(roi);
        else
            distance = moda(roi, roi8);
        //Dibujar
        rectangle(image[0], rect, Scalar(0,255,0), 3);
        Point anchor(xi+10, yi+18);
        putText(image[0], to_string(distance/1000), anchor, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
        imshow(distance_window, image[0]);
    }

}

// disp_src Matriz de disparidad thresholdeada, dst matriz de destino
void calculate_blobs(Mat binary_map_src, vector<Rect> &objects, Mat& dst, int min_area=80);

//Calculara la distancia a los objetos y la mostrara
void calculate_depth(Mat disp, Mat disp8, Mat &dst, vector<Rect> objects);

//Leeme: Hay que meter 3 argumentos que seran los numeros de las camaras
void readme(){
    cout << "Uso: ./stereo_matchBM <camera0> <camera1> [-c calib_file] [-d 1|2]" << endl;
}

int main(int argc, char **argv){
    //Pulsar r para resumir la captura de video
    bool rend =  true, go = true;

    if(argc < 3){ readme(); return -1; }

    args(argc, argv);

    FileStorage fs(calib_filename, FileStorage::READ);
    if(!fs.isOpened()) { cout << "ERROR! El fichero de calibracion no se pudo abrir" << endl; readme(); return -1; }
    calibData.read(fs);
    fs.release();

    VideoCapture cap1(camera1);
    VideoCapture cap2(camera2);
    if(!(cap1.isOpened() || cap2.isOpened())) { cout << "ERROR! La camara no puso ser abierta" << endl; readme(); return -1; }
    //cap1.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
    //cout << "FOURCC " << cap1.get(CV_CAP_PROP_FOURCC) << endl;
    //cap1.set(CV_CAP_PROP_FPS, 30);
    //cout << "FPS " << cap1.get(CV_CAP_PROP_FPS);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, calibData.frame_width);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, calibData.frame_height);
    //cap2.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
    //cap2.set(CV_CAP_PROP_FPS, 30);
    //cout << "FPS " << cap2.get(CV_CAP_PROP_FPS);
    cap2.set(CV_CAP_PROP_FRAME_WIDTH, calibData.frame_width);
    cap2.set(CV_CAP_PROP_FRAME_HEIGHT, calibData.frame_height);

    Mat imageU[2], imageUG[2], depth_map;
    Mat map1x, map1y, map2x, map2y;
    cap1 >> image[0];
    StereoBM bm;
    bm.state->roi1 = calibData.roi[0];
    bm.state->roi2 = calibData.roi[1];

    cout << "Para Min Disparity sea negativo pulsa m" << endl;

    //Trackbars
    namedWindow(trackWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(disparityWindow, CV_WINDOW_AUTOSIZE);
    createTrackbar(pre_filter_size_trackbar, trackWindow, &preFilterSize, 255, pre_filter_size_callback);
    createTrackbar("Pre-Filter Cap", trackWindow, &preFilterCap, 63);
    createTrackbar("SAD Window Size", trackWindow, &sadWindowsize, 255, sad_window_size_callback);
    createTrackbar("Min Disp", trackWindow, &minDisparity, 100);
    createTrackbar("Num Disp *16", trackWindow, &numberOfDisparities, 16);
    createTrackbar("Texture Threshold", trackWindow, &textureThreshold, 1000);
    createTrackbar("Uniqueness Ratio", trackWindow, &uniqnessRatio, 255);
    createTrackbar("Speckle Win Size", trackWindow, &speckleWindowSize, 200);
    createTrackbar("Speckle Range", trackWindow, &speckleRange, 100);
    //createTrackbar("disp12MaxDiff", trackWindow, &disp12MaxDiff, 100);
    createTrackbar("threshold", trackWindow, &thresholdRange, 255);
    createTrackbar(min_blob_area_trackbar, trackWindow, &min_blob_area, 1000);

    //Callbacks
    if(DEBUG) setMouseCallback(disparityWindow, disp_window_mouse_callback, &depth_map);

    //Rectificar camara
    initUndistortRectifyMap(calibData.CM[0], calibData.D[0],calibData.r[0], calibData.P[0], image[0].size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(calibData.CM[1], calibData.D[1],calibData.r[1], calibData.P[1], image[0].size(), CV_32FC1, map2x, map2y);
    //Se hara threshold para calcular el mapa binario
    Mat dispT, blobs;//(480, 640, CV_8UC3);
    while(go){
        double t = (double) getTickCount();
        if(rend){
            cap1 >> image[0];
            cap2 >> image[1];
        }
        //Parametros BM
        bm.state->SADWindowSize = sadWindowsize;
        bm.state->numberOfDisparities = numberOfDisparities*16;
        bm.state->preFilterSize = preFilterSize;
        bm.state->preFilterCap = preFilterCap;
        bm.state->minDisparity = minDisparityNeg ? -1*minDisparity : minDisparity;
        bm.state->textureThreshold = textureThreshold;
        bm.state->uniquenessRatio = uniqnessRatio;
        bm.state->speckleWindowSize = speckleWindowSize;
        bm.state->speckleRange = speckleRange;
        bm.state->disp12MaxDiff = disp12MaxDiff;
        
        //Hacemos remap
        remap(image[0], imageU[0], map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(image[1], imageU[1], map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        //Cambiamos a escala de grises
        cvtColor(imageU[0], imageUG[0], CV_BGR2GRAY);
        cvtColor(imageU[1], imageUG[1], CV_BGR2GRAY);
        //Calculo del mapa de disparidad
        bm(imageUG[0], imageUG[1], disp, CV_32F); //Tiene mas parametros
        //Es necesario normalizar el mapa de disparidad
        normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
        threshold(disp8, dispT, thresholdRange, 255, 0);

        vector<Rect> objects;
        if(rend) calculate_blobs(dispT, objects, blobs, min_blob_area);
        
        //Calculo de la coordenada Z de todos los puntos
        //TODO: Cambiar esto por algo mas logico y eficiente
        //if(DEBUG) reprojectImageTo3D(disp, depth_map, calibData.Q);

        if(rend) calculate_depth(disp, disp8, blobs, objects);

        imshow("Left", imageU[0]);
        imshow("Right", imageU[1]);
        imshow("Mapa Binario", dispT);
        imshow(disparityWindow, disp8);
        imshow("Blobs", blobs);
        switch (waitKey(1)){
            case 1048603:
                go = false;
                break;
            case 1048608:
                imwrite("capL.png", image[0]); 
                imwrite("capR.png", image[1]); 
                imwrite("capUL.png", imageU[0]); 
                imwrite("capUR.png", imageU[1]); 
                imwrite("disp.png", disp8);
                imwrite("dispT.png", dispT);
                imwrite("blobs.png", blobs);
                cout << "Captura izq y der guardada" << endl;
                break;
            case 1048690:
                rend = !rend;
                break;
	    case 1048676:
                if(distance_method == 1){
                    distance_method = 2;
                    cout << "Metodo de distancia Moda" << endl;
                }else{
                    distance_method = 1;
                    cout << "Metodo de distancia Media" << endl;
                }
                break;
            case 1048685:
                minDisparityNeg = !minDisparityNeg;
                if(minDisparityNeg) cout << "Min Disp es negativo" << endl;
                else cout << "Min Disp es positivo" << endl;
                break;
            default:
                break;
        }
    if(rend) cout << "Tiempo ciclo: " << ((double)getTickCount() - t)/getTickFrequency() << "s" << endl;
    }
}

void calculate_blobs(Mat binary_map_src, vector<Rect> &objects, Mat& dst, int min_area){
    //Buena y mala idea. rellena huecos, pero causa confusion con otros objetos
    //dilate(binary_map_src, binary_map_src, Mat());
    erode(binary_map_src, binary_map_src, Mat());
    CBlobResult blobs(binary_map_src);
    CBlob *blob;

    //Filtrar blobs por area
    CBlobResult blobs_filtered;
    blobs.Filter(blobs_filtered, FilterAction::FLT_EXCLUDE,  CBlobGetArea(), FilterCondition::FLT_LESS, min_area);

    //dst(binary_map_src.size().height, binary_map_src.size().width, CV_8UC3);
    dst.create(binary_map_src.size(), CV_8UC3);
    dst = Scalar(0);
    for(int i=0; i < blobs_filtered.GetNumBlobs(); i++){
        blob = blobs_filtered.GetBlob(i);
        blob->FillBlob(dst, Scalar(rand() % 255, rand() % 255, rand() % 255));
        //if(blob->MaxY() >= 355){ /*Para 480*/
        if(blob->MaxY() >= 175){ /*Para 240*/
            Rect blob_rect = blob->GetBoundingBox();
            objects.push_back(blob_rect);
            rectangle(dst, blob_rect, Scalar(0,255,0), 3);
        }
    }
}

double media(Mat roi){
    Mat depth;
    vector<Mat> depthC;
    reprojectImageTo3D(roi, depth, calibData.Q);
    split(depth, depthC);
    float *p; 
    double sum=0;
    int cont =0;
    for(int k=0;  k < depthC[2].rows; k++){
        p = depthC[2].ptr<float>(k);
        for(int j=0; j < depthC[2].cols; j++){
            if(p[j] > 0){
                sum += p[j];
                cont++;
            }
        }
    }
    return sum/cont;
}

double moda(Mat roi, Mat roi8){
    Mat hist, depth;
    int histSize = 255;
    float range[] = {1, 256};
    const float *histRange = {range};
    calcHist(&roi8, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    Point max_loc;
    minMaxLoc(hist, NULL, NULL, NULL, &max_loc);
    
    uchar *p;
    int x,y;
    bool found = false;
    for(int k=0; k < roi8.rows; k++){
        p = roi8.ptr<uchar>(k);
        for(int j=0; j < roi8.cols; j++){
            if(p[j] == max_loc.y+1) { 
                x=j;
                y=k;
                found=true;
                break;
            }
        }
        if(found) break;
    }
    
    //Conseguimos la disparidad de la moda
    Mat rep_roi(roi, Rect(Point(x,y), Size(1,1)));
    reprojectImageTo3D(rep_roi, depth, calibData.Q);
    return depth.at<Vec3f>(0,0)[2];
}

void calculate_depth(Mat disp, Mat disp8, Mat &dst, vector<Rect> objects){
    Mat depth;
    vector<Mat> depthC;
    double distance;
    for(int i=0; i < objects.size(); i++){
        Mat roi(disp, objects[i]);

        if(distance_method == 1){
            distance = media(roi);
        }else{
            //Se calcula el histograma para ver cual se repit mas
            Mat roi8(disp8, objects[i]);
            distance = moda(roi, roi8); 
        }

        Point anchor(objects[i].x+10, objects[i].y+18);
        putText(dst, to_string(distance/1000), anchor, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    }
}
