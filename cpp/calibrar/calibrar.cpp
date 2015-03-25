/** Este programa servira para calibrar una unica camara **/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    int numFotos;
    int numCornersHor;
    int numCornersVer;
    int cornerSize;
    // Matrices importantes
    Mat intrinsics = Mat::eye(3, 3, CV_64F);
    Mat distCoef = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvec;
    vector<Mat> tvec;
    String cameraWindow = "Camara";
    String goodCameraWindow = "Camara calibrada";
    namedWindow(cameraWindow, CV_WINDOW_AUTOSIZE);
    namedWindow(goodCameraWindow, CV_WINDOW_AUTOSIZE);
    cout << "Introduce el numero de esquinas horizontales: "; cin >> numCornersHor;
    cout << "Introduce el numero de esquinas verticales: "; cin >> numCornersVer;
    cout << "Introduce el tamaño del cuadrado negro: "; cin >> cornerSize;
    cout << "Introduce el numero de fotos a hacer: "; cin >> numFotos;
    VideoCapture cap(0);
    if(!cap.isOpened()){ cout << "La camara no se pudo abrir" << endl; return -1;}
    cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
    //Vectores que guardan los puntos en 3D y 2D correspondientes a las esquinas del chess en cada imagen
    vector<vector<Point3f> > objectPoints;
    vector<vector<Point2f> > imagePoints;
    //Tamaño del tablero
    Size chess_size(numCornersHor, numCornersVer);
    //Calculo de las coordenada 3D del ajedrez. Se tomara como origen de coordenadas la primera esquina izquierda del patron. Con Z=0
    vector<Point3f> obj;
    for(int i = 0; i < numCornersVer; i++){
        for(int j=0; j < numCornersHor; j++){
            obj.push_back(Point3f(float(j*cornerSize), float(i*cornerSize), 0.0f));
        }
    }
    cout << "Calculadas coordenadas 3D" << endl;
    int capturas = 0;
    Mat image;
    Mat gray_image;
    //Donde se guardaran las esquinas encontradas por el metodo findChessboardCorners
    vector<Point2f> corners;
    while(capturas < numFotos){
        cap >> image;
        //Metodo que busca las esquinas y las guarda en corners
        bool found = findChessboardCorners(image, chess_size, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        //Si se encuentran las esquinas se hace una mejora en estas y se dibujan en pantalla
        if(found){
            cvtColor(image, gray_image, CV_BGR2GRAY);
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(image, chess_size, corners, found);
        }
        imshow(cameraWindow, image);
        int key = waitKey(1);
        // Si se pulsa ESC se saldra del programa
        if(key == 1048603)return 0;
        // Si pulsamos espacio guardaremos la captura
        if(key == 1048608 && found){
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);
            capturas++;
            cout << "Captura guardada!! (" << capturas << "/" << numFotos << ")" << endl;
        }
    }
    //Calibramos la camara
    calibrateCamera(objectPoints, imagePoints, image.size(), intrinsics, distCoef, rvec, tvec);
    Mat imagenCorrecta;
    while(true){
        cap >> image;
        undistort(image, imagenCorrecta, intrinsics, distCoef);
        imshow(cameraWindow, image);
        imshow(goodCameraWindow, imagenCorrecta);
        if(waitKey(1) == 1048603) break;
    }
    cap.release();
    return 0;
}
