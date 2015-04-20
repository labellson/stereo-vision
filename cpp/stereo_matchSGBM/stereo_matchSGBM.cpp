/** Calculo de un mapa de disparidad mediante el algoritmo Block matching **/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencvblobslib/BlobResult.h>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobOperators.h>
#include "SCalibData.h"

using namespace cv;
using namespace std;

// disp_src Matriz de disparidad thresholdeada, dst matriz de destino
void calculate_blobs(Mat binary_map_src, Mat& dst);

int main(int argc, char **argv){
    //Pulsar r para resumir la captura de video
    bool rend =  true;
    SCalibData calibData;
    FileStorage fs("../resources/stereo_calib.yml", FileStorage::READ);
    calibData.read(fs);
    fs.release();
    VideoCapture cap1(0);
    VideoCapture cap2(1);
    Mat image[2], imageU[2], imageUG[2], disp, disp8;
    Mat map1x, map1y, map2x, map2y;
    cap1 >> image[0];
    int cn = image[0].channels();
    StereoSGBM sgbm;
    //Parametros inicio
    /*sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = ((image[0].size().width/8) + 15) & -16; //Probar variable
    sgbm.disp12MaxDiff = 1; //Diferencia maxima permitida en el check de disparidad. -1 desactiva el check
    sgbm.preFilterCap = 63;
    sgbm.speckleRange = 32;
    sgbm.fullDP = false;*/
    //Variables sgbm
    int sad_window_size = 1; //Normalmente entre 3..11
    int uniqueness_ratio = 5; //Normalmente entre 5..15
    int speckle_window_size = 50; //Filtrar ruido. Normalmente entre 50..200
    //Variables prueba
    int minDisparity = 0;
    int disp12MaxDiff = 1;
    int preFilterCap = 63;
    int speckleRange = 32;
    int thresholdRange = 0;
    //Trackbars
    String trackWindow = "Settings";
    namedWindow(trackWindow, CV_WINDOW_AUTOSIZE);
    createTrackbar("SADWindowSize 3..11", trackWindow, &sad_window_size, 50);
    createTrackbar("uniquenessRatio 5..15", trackWindow, &uniqueness_ratio, 20);
    createTrackbar("speckleWindowSize 50..200", trackWindow, &speckle_window_size, 200);
    //Trackbars prueba
    createTrackbar("minDisparity", trackWindow, &minDisparity, 20);
    createTrackbar("disp12MaxDiff", trackWindow, &disp12MaxDiff, 12);
    createTrackbar("preFilterCap", trackWindow, &preFilterCap, 100);
    createTrackbar("speckleRange", trackWindow, &speckleRange, 128);
    createTrackbar("threshold", trackWindow, &thresholdRange, 255);
    //Rectificar camara
    initUndistortRectifyMap(calibData.CM[0], calibData.D[0],calibData.r[0], calibData.P[0], image[0].size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(calibData.CM[1], calibData.D[1],calibData.r[1], calibData.P[1], image[0].size(), CV_32FC1, map2x, map2y);
    //Se utilazaran los roi para el mapa de disparidad
    int roi_x = calibData.roi[0].x > calibData.roi[1].x ? calibData.roi[0].x : calibData.roi[1].y;
    int roi_y = calibData.roi[0].y > calibData.roi[1].y ? calibData.roi[0].y : calibData.roi[1].y;
    int roi_width = calibData.roi[0].width < calibData.roi[1].width ? calibData.roi[0].width : calibData.roi[1].width;
    int roi_height = calibData.roi[0].height < calibData.roi[1].height ? calibData.roi[0].height : calibData.roi[1].height;
    Rect roi(roi_x, roi_y, roi_width, roi_height);
    //Se hara threshold para calcular el mapa binario
    while(true){
    Mat dispT, blobs;
        if(rend){
            cap1 >> image[0];
            cap2 >> image[1];
        }
        //Parametros SGBM
        sgbm.P1 = 8*cn*sad_window_size*sad_window_size;
        sgbm.P2 = 32*cn*sad_window_size*sad_window_size;
        sgbm.SADWindowSize = sad_window_size;
        sgbm.uniquenessRatio = uniqueness_ratio;
        sgbm.speckleWindowSize = speckle_window_size;
        //Parametros de prueba
        sgbm.minDisparity = minDisparity;
        sgbm.numberOfDisparities = ((image[0].size().width/8) + 15) & -16; //Probar variable
        sgbm.disp12MaxDiff = disp12MaxDiff; //Diferencia maxima permitida en el check de disparidad. -1 desactiva el check
        sgbm.preFilterCap = preFilterCap;
        sgbm.speckleRange = speckleRange;
        sgbm.fullDP = false;
        //remap(image[0], imageU[0], calibData.map[0][0], calibData.map[0][1], INTER_LINEAR, BORDER_CONSTANT, Scalar());
        //remap(image[1], imageU[1], calibData.map[1][0], calibData.map[1][1], INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(image[0], imageU[0], map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(image[1], imageU[1], map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        //Cambiamos a escala de grises
        cvtColor(imageU[0], imageUG[0], CV_BGR2GRAY);
        cvtColor(imageU[1], imageUG[1], CV_BGR2GRAY);
        //Calculo del mapa de disparidad
        sgbm(imageUG[0], imageUG[1], disp); //Tiene mas parametros
        //Es necesario normalizar el mapa de disparidad
        normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
        //Mat disp8roi(disp8, roi);
        threshold(disp8, dispT, thresholdRange, 255, 0);
        calculate_blobs(dispT, blobs);
        imshow("Left", imageU[0]);
        imshow("Right", imageU[1]);
        imshow("Mapa Binario", dispT);
        imshow("Disparidad", disp8);
        imshow("Blobs", blobs);
        if(waitKey(1) == 1048603) break;
        if(waitKey(1) == 1048690) rend = !rend;
    }
}

void calculate_blobs(Mat binary_map_src, Mat& dst){
    CBlobResult blobs(binary_map_src);
    CBlob *blob;
    int min_area = 80;

    CBlobResult blobs_filtered;
    blobs.Filter(blobs_filtered, FilterAction::FLT_EXCLUDE,  CBlobGetArea(), FilterCondition::FLT_LESS, min_area);

    dst = cvCreateMat(binary_map_src.size().height, binary_map_src.size().width, CV_8UC3);
    for(int i=0; i < blobs_filtered.GetNumBlobs(); i++){
        blob = blobs_filtered.GetBlob(i);
        blob->FillBlob(dst, Scalar(rand() % 255, rand() % 255, rand() % 255));
    }
}
