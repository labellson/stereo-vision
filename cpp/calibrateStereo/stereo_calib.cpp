/** Este programa servira para calibrar una unica camara **/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include "SCalibData.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    int numFotos;
    int numCornersHor;
    int numCornersVer;
    int cornerSize;
    int frame_width = argc < 3 ? 640 : stoi(argv[1]);
    int frame_height = argc < 3 ? 480 : stoi(argv[2]);
    // Matrices importantes
    Mat cameraMatLeft = Mat::eye(3, 3, CV_64F);
    Mat cameraMatRight = Mat::eye(3, 3, CV_64F);
    Mat distCoefLeft = Mat::zeros(8, 1, CV_64F);
    Mat distCoefRight = Mat::zeros(8, 1, CV_64F);
    // Rotacion, Traslacion, Fundamental, Esencial
    Mat R, T, F, E;
    String left = "Izquierda";
    String right = "Derecha";
    namedWindow(left, CV_WINDOW_AUTOSIZE);
    namedWindow(right, CV_WINDOW_AUTOSIZE);
    cout << "Introduce el numero de esquinas horizontales: "; cin >> numCornersHor;
    cout << "Introduce el numero de esquinas verticales: "; cin >> numCornersVer;
    cout << "Introduce el tamaño del cuadrado negro: "; cin >> cornerSize;
    cout << "Introduce el numero de fotos a hacer: "; cin >> numFotos;
    int camIzq, camDer;
    cout << "Introduce la camara izquierda: "; cin >> camIzq;
    cout << "Introduce la camara derecha: "; cin >> camDer;
    VideoCapture capLeft(camIzq);
    capLeft.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);
    capLeft.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
    VideoCapture capRight(camDer);
    capRight.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);
    capRight.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
    if(!capLeft.isOpened() || !capRight.isOpened()){ cout << "La camara no se pudo abrir" << endl; return -1;}
    //Vectores que guardan los puntos en 3D y 2D correspondientes a las esquinas del chess en cada imagen
    vector<vector<Point3f> > objectPoints;
    vector<vector<Point2f> > imagePointsLeft, imagePointsRight;
    //Tamaño del tablero
    Size chess_size(numCornersHor, numCornersVer);
    //Calculo de las coordenada 3D del ajedrez. Se tomara como origen de coordenadas la primera esquina izquierda del patron. Con Z=0
    vector<Point3f> obj;
    for(int i = 0; i < numCornersVer; i++){
        for(int j=0; j < numCornersHor; j++){
            obj.push_back(Point3f(float(j*cornerSize), float(i*cornerSize), 0.0f));
        }
    }
    int capturas = 0;
    Mat imageLeft, imageRight;
    Mat gray_imageLeft, gray_imageRight;
    //Donde se guardaran las esquinas encontradas por el metodo findChessboardCorners
    vector<Point2f> cornersLeft, cornersRight;
    while(capturas < numFotos){
        capLeft >> imageLeft;
        capRight >> imageRight;
        //Metodo que busca las esquinas y las guarda en corners
        bool foundLeft = findChessboardCorners(imageLeft, chess_size, cornersLeft, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        bool foundRight = findChessboardCorners(imageRight, chess_size, cornersRight, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        //Si se encuentran las esquinas se hace una mejora en estas y se dibujan en pantalla
        if(foundLeft && foundRight){
            cvtColor(imageLeft, gray_imageLeft, CV_BGR2GRAY);
            cvtColor(imageRight, gray_imageRight, CV_BGR2GRAY);
            cornerSubPix(gray_imageLeft, cornersLeft, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cornerSubPix(gray_imageRight, cornersRight, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(imageLeft, chess_size, cornersLeft, foundLeft);
            drawChessboardCorners(imageRight, chess_size, cornersRight, foundRight);
        }
        imshow(left, imageLeft);
        imshow(right, imageRight);
        int key = waitKey(1);
        // Si se pulsa ESC se saldra del programa
        if(key == 1048603)return 0;
        // Si pulsamos espacio guardaremos la captura
        if(key == 1048608 && foundLeft && foundRight){
            imagePointsLeft.push_back(cornersLeft);
            imagePointsRight.push_back(cornersRight);
            objectPoints.push_back(obj);
            capturas++;
            cout << "Captura guardada!! (" << capturas << "/" << numFotos << ")" << endl;
        }
    }
    //Calibramos el conjunto bifocal
    stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight, cameraMatLeft, distCoefLeft, cameraMatRight, distCoefRight, imageLeft.size(), R, T, E, F, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6), CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST);
    // Matriz de rectificacion (Rotacion) izquierda, Mismo derecha, Matriz proyeccion izq, Matriz proyeccion der, Matriz disparidad a profundidad
    Mat R1, R2, P1, P2, Q; 
    Rect roi[2];
    stereoRectify(cameraMatLeft, distCoefLeft, cameraMatRight, distCoefRight, imageLeft.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, Size(0,0), &roi[0], &roi[1]);
    //Matrices de mapeo para remap
    Mat map1x, map1y, map2x, map2y;
    //Imagenes rectificadas y sin distorsion
    Mat imageU[2]; 
    double sf = 600./MAX(imageLeft.size().width, imageLeft.size().height);
    int w = cvRound(imageLeft.size().width*sf), h = cvRound(imageLeft.size().height*sf);
    Mat canvas(h, w*2, CV_8UC3);
    initUndistortRectifyMap(cameraMatLeft, distCoefLeft, R1, P1, imageLeft.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cameraMatRight, distCoefRight, R2, P2, imageRight.size(), CV_32FC1, map2x, map2y);
    //Lo guardamos todo en un fichero yaml
    FileStorage fs("stereo_calib_"+to_string(frame_height)+".yml", FileStorage::WRITE);
    SCalibData calibData(cameraMatLeft, cameraMatRight, distCoefLeft, distCoefRight, R, T, E, F, R1, R2, P1, P2, Q, roi[0], roi[1], frame_width, frame_height);
    calibData.write(fs);
    fs.release();
    while(true){
        capLeft >> imageLeft;
        capRight >> imageRight;
        remap(imageLeft, imageU[0], map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(imageRight, imageU[1], map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        for( int k=0; k < 2; k++){
            Mat canvasPart(canvas, Rect(w*k, 0, w, h));
            rectangle(imageU[k], roi[k], Scalar(0, 255, 0), 4);
            resize(imageU[k], canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
        }
        for (int j=0; j < canvas.rows; j+=16){
            line(canvas, Point(0,j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        }
        imshow(left, imageLeft);
        imshow(right, imageRight);
        imshow("Rectificado", canvas);
        //imshow("LeftU", imageU[0]);
        //imshow("RightU", imageU[1]);
        if(waitKey(1) == 1048603) break;
    }
    return 0;
}
