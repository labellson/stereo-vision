/** Calculo de un mapa de disparidad mediante el algoritmo Block matching **/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <queue>
#include <string>
#include <opencvblobslib/BlobResult.h>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobOperators.h>
#include <omp.h>
#include "SCalibData.h"

#define DEBUG 0

using namespace cv;
using namespace std;

//Variables y Buffers
// Elementos
// pre_proccess_buff: [image0, image1]
// disp_buff: [disp, disp8, dispT] disp, disp_normalizada, disp_thresholdeada
queue<Mat*> pre_proccess_buff, disp_buff;

//Variables importante
Mat *image, *imageU, depth_map;
Mat map1x, map1y, map2x, map2y;
StereoBM bm;

//Strings Trackbars y WinNames
String trackWindow = "Settings", disparityWindow = "Disparidad";
String leftWindow = "Left", rightWindow = "Right";
String binaryWindow = "Mapa Binario", blobWindow = "Blobs";
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
int disp12MaxDiff = 1;
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

static void disp_window_mouse_callback(int event, int x, int y, int flags, void* userdata){
    if(event != EVENT_LBUTTONDOWN) return;
    cout << "x: " << x << " y: " << y << endl;
    Mat* depth_map = (Mat*) userdata;
    Vec3f vec = depth_map->at<Vec3f>(y,x);
    cout << "x: " << vec[0] << " y: " << vec[1] << " z: " << vec[2] << endl;
}

// disp_src Matriz de disparidad thresholdeada, dst matriz de destino
void calculate_blobs(Mat binary_map_src, vector<Rect> &objects, Mat& dst, int min_area=80);

double media(Mat roi);

double moda(Mat roi, Mat roi8);

//Calculara la distancia a los objetos y la mostrara
void calculate_depth(Mat disp, Mat disp8, Mat &dst, vector<Rect> objects);

//Hara el preprocesado de la imagen, una vez capturada
void pre_proccess_images(Mat &image0, Mat &image1);

//Calcula el mapa de Disparidad
void get_disparity_map(Mat *imageUG);

//Leeme: Hay que meter 2 argumentos que seran los numeros de las camaras
void readme(){
    cout << "Uso: ./stereo_matchBM <camera0> <camera1> [-c calib_file] [-d 1|2]" << endl;
}

int main(int argc, char **argv){
    //Pulsar r para resumir la captura de video
    bool rend =  true, go = true;
    
    image = new Mat[2];
    imageU = new Mat[2];

    if(argc < 3){ readme(); return -1; }

    args(argc, argv);

    FileStorage fs(calib_filename, FileStorage::READ);
    if(!fs.isOpened()) { cout << "ERROR! El fichero de calibracion no se pudo abrir" << endl; readme(); return -1; }
    calibData.read(fs);
    fs.release();

    //Inicializacion
    VideoCapture cap1(camera1);
    VideoCapture cap2(camera2);
    if(!(cap1.isOpened() || cap2.isOpened())) { cout << "ERROR! La camara no puso ser abierta" << endl; readme(); return -1; }
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, calibData.frame_width);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, calibData.frame_height);
    cap2.set(CV_CAP_PROP_FRAME_WIDTH, calibData.frame_width);
    cap2.set(CV_CAP_PROP_FRAME_HEIGHT, calibData.frame_height);

    cap1 >> image[0];
    bm.state->roi1 = calibData.roi[0];
    bm.state->roi2 = calibData.roi[1];

    cout << "Para Min Disparity sea negativo pulsa m" << endl;

    //Trackbars
    namedWindow(trackWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(disparityWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(leftWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(rightWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(binaryWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(blobWindow, CV_WINDOW_AUTOSIZE);
    createTrackbar(pre_filter_size_trackbar, trackWindow, &preFilterSize, 255, pre_filter_size_callback);
    createTrackbar("Pre-Filter Cap", trackWindow, &preFilterCap, 63);
    createTrackbar("SAD Window Size", trackWindow, &sadWindowsize, 255, sad_window_size_callback);
    createTrackbar("Min Disp", trackWindow, &minDisparity, 100);
    createTrackbar("Num Disp *16", trackWindow, &numberOfDisparities, 16);
    createTrackbar("Texture Threshold", trackWindow, &textureThreshold, 1000);
    createTrackbar("Uniqueness Ratio", trackWindow, &uniqnessRatio, 255);
    createTrackbar("Speckle Win Size", trackWindow, &speckleWindowSize, 200);
    createTrackbar("Speckle Range", trackWindow, &speckleRange, 100);
    createTrackbar("disp12MaxDiff", trackWindow, &disp12MaxDiff, 100);
    createTrackbar("threshold", trackWindow, &thresholdRange, 255);
    createTrackbar(min_blob_area_trackbar, trackWindow, &min_blob_area, 1000);

    //Callbacks
    if(DEBUG) setMouseCallback(disparityWindow, disp_window_mouse_callback, &depth_map);

    //Rectificar camara
    initUndistortRectifyMap(calibData.CM[0], calibData.D[0],calibData.r[0], calibData.P[0], image[0].size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(calibData.CM[1], calibData.D[1],calibData.r[1], calibData.P[1], image[0].size(), CV_32FC1, map2x, map2y);
    //Se hara threshold para calcular el mapa binario
    Mat blobs;//(480, 640, CV_8UC3);
    double t;
    int threads = 0; //threads maximos calculando disparidades 3
    #pragma omp parallel shared(pre_proccess_buff, disp_buff, t, go, rend), num_threads(4)
    {
        #pragma omp single nowait
        {
        while(go){
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

            //Captura de las imagenes
            if(rend && pre_proccess_buff.size() < 4){
                cap1 >> image[0];
                cap2 >> image[1];

                //Rectificado y preprocesado de las imagenes
                pre_proccess_images(image[0], image[1]);
            }
        }
        }

        #pragma omp single
        {
        while(go){
        //Si la cola esta vacia, es que es la primera ejecucion
        if(!pre_proccess_buff.empty() && threads < 3){
            #pragma omp task
            {
            #pragma omp atomic
            threads++; //Incremento del contador de threads en uso
            //Seccion critica, no se puede leer/escribir a la vez
            Mat *proc_par;
            bool empty;
            #pragma omp critical(pre_proccess_buff)
            {
            //Se vuelve a mirar si el buffer tiene imagenes, para evitar falsos positivos
            empty = pre_proccess_buff.empty();
            if(!empty){
                proc_par = pre_proccess_buff.front();
                //Limpiar la cola
                pre_proccess_buff.pop();
            }
            }

            if(!empty){
                //Calculo del mapa de disparidad
                get_disparity_map(proc_par);
                
                //Mas seccion critica
                Mat *disp_array;
                #pragma omp critical(disp_buff)
                {
				disp_array = disp_buff.front();
				disp_buff.pop();
                }

				vector<Rect> objects;
				Mat blobs1;
				if(rend) calculate_blobs(disp_array[2], objects, blobs1, min_blob_area);
				
				//Calculo de la coordenada Z de todos los puntos
				//TODO: Cambiar esto por algo mas logico y eficiente
				if(DEBUG) reprojectImageTo3D(disp_array[0], depth_map, calibData.Q);

				if(rend) calculate_depth(disp_array[0], disp_array[1], blobs1, objects);

				if(rend){
					#pragma omp critical(window)
					{
					imshow(leftWindow, proc_par[0]);
					imshow(rightWindow, proc_par[1]);
					imshow(binaryWindow, disp_array[2]);
					imshow(disparityWindow, disp_array[1]);
					imshow(blobWindow, blobs1);
					}
				}
                //Necesario limpiar esta parte de la memoria, para evitar memory leaks
                delete[] proc_par;
                delete[] disp_array;
				#pragma omp critical(tick)
				{
				cout << "Tiempo ciclo: " << ((double)getTickCount() - t)/getTickFrequency() << "s" << endl;
				t = (double) getTickCount();
                }
            }
            #pragma omp atomic
            threads--;
        }
        }
        switch (waitKey(1)){
            case 1048603:
                go = false;
                break;
            case 1048608:
                imwrite("capL.png", imageU[0]); 
                imwrite("capR.png", imageU[1]); 
                cout << "Captura izq y der guardada" << endl;
                break;
            case 1048690:
                rend = !rend;
                break;
            case 1048685:
                minDisparityNeg = !minDisparityNeg;
                if(minDisparityNeg) cout << "Min Disp es negativo" << endl;
                else cout << "Min Disp es positivo" << endl;
                break;
            default:
                break;
        }
        }
        }
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
/*void calculate_depth(Mat disp, Mat &dst, vector<Rect> objects){
    Mat depth;
    vector<Mat> depthC;
    Vec3f vec;
    for(int i=0; i < objects.size(); i++){
        Mat roi(disp, objects[i]);
        reprojectImageTo3D(roi, depth, calibData.Q);

        split(depth, depthC);
        float *p, sum=0;
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

        Point anchor(objects[i].x+10, objects[i].y+18);
        putText(dst, to_string(sum/cont/1000), anchor, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    }
}*/

void pre_proccess_images(Mat &image0, Mat &image1){
    Mat *imageP = new Mat[2];
    
    //Hacemos remap
    remap(image0, imageU[0], map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    imageP[0] = imageU[0];
    remap(image1, imageU[1], map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    imageP[1] = imageU[1];

    //Cambiamos a escala de grises
    cvtColor(imageP[0], imageP[0], CV_BGR2GRAY);
    cvtColor(imageP[1], imageP[1], CV_BGR2GRAY);

    //A la cola!!
    #pragma omp critical(pre_proccess_buff)
    {
    pre_proccess_buff.push(imageP);
    }
}

void get_disparity_map(Mat* imageUG){
    Mat disp, disp8, dispT;
    Mat *disparidades = new Mat[3];
    
    //Calculo del mapa de disparidad
	#pragma omp critical(bm)
    bm(imageUG[0], imageUG[1], disp, CV_32F); //Tiene mas parametros
    disparidades[0] = disp;
    
    //Es necesario normalizar el mapa de disparidad
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    disparidades[1] = disp8;

    //Thresholdeamos el mapa de disparidad
    threshold(disp8, dispT, thresholdRange, 255, 0);
    disparidades[2] = dispT;

    //A la cola!!
    #pragma omp critical(disp_buff)
    {
    disp_buff.push(disparidades);
    }
}
