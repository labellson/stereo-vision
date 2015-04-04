/** Calculo de un mapa de disparidad mediante el algoritmo Block matching **/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "SCalibData.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
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
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = ((image[0].size().width/8) + 15) & -16; //Probar variable
    sgbm.disp12MaxDiff = 1; //Diferencia maxima permitida en el check de disparidad. -1 desactiva el check
    sgbm.preFilterCap = 63;
    sgbm.speckleRange = 32;
    sgbm.fullDP = false;
    //Variables sgbm
    int sad_window_size = 1; //Normalmente entre 3..11
    int uniqueness_ratio = 5; //Normalmente entre 5..15
    int speckle_window_size = 50; //Filtrar ruido. Normalmente entre 50..200
    //Trackbars
    String trackWindow = "Settings";
    namedWindow(trackWindow, CV_WINDOW_AUTOSIZE);
    createTrackbar("SADWindowSize 3..11", trackWindow, &sad_window_size, 50);
    createTrackbar("uniquenessRatio 5..15", trackWindow, &uniqueness_ratio, 20);
    createTrackbar("speckleWindowSize 50..200", trackWindow, &speckle_window_size, 200);
    //Rectificar camara
    initUndistortRectifyMap(calibData.CM[0], calibData.D[0],calibData.r[0], calibData.P[0], image[0].size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(calibData.CM[1], calibData.D[1],calibData.r[1], calibData.P[1], image[0].size(), CV_32FC1, map2x, map2y);
    while(true){
        cap1 >> image[0];
        cap2 >> image[1];
        //Parametros SGBM
        sgbm.P1 = 8*cn*sad_window_size*sad_window_size;
        sgbm.P2 = 32*cn*sad_window_size*sad_window_size;
        sgbm.SADWindowSize = sad_window_size;
        sgbm.uniquenessRatio = uniqueness_ratio;
        sgbm.speckleWindowSize = speckle_window_size;
        //remap(image[0], imageU[0], calibData.map[0][0], calibData.map[0][1], INTER_LINEAR, BORDER_CONSTANT, Scalar());
        //remap(image[1], imageU[1], calibData.map[1][0], calibData.map[1][1], INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(image[0], imageU[0], map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(image[1], imageU[1], map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        //cvtColor(imageU[0], imageUG[0], CV_BGR2GRAY);
        //cvtColor(imageU[1], imageUG[1], CV_BGR2GRAY);
        //sgbm(imageUG[0], imageUG[1], disp); //Tiene mas parametros
        sgbm(imageU[0], imageU[1], disp); //Tiene mas parametros
        normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
        imshow("Left", imageU[0]);
        imshow("Right", imageU[1]);
        imshow("Disparidad", disp8);
        if(waitKey(1) == 1048603) break;
    }
}
