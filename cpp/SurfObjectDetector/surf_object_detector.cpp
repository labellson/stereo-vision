#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <string>

using namespace cv;
using namespace std;

void readme();

//Args str
String bw_arg = "-bw", min_match_arg = "-m", cam_arg = "-c";
String help_arg = "-h";

//Args
bool bw = false;
int min_match = 7, cam = 0;

void args(int argc, char **argv);

void man();

void surf(Mat &img, vector<KeyPoint> &kp, Mat &des);

void surfObject(Mat &img_obj, vector<KeyPoint> &kp_obj, Mat &des_obj, vector<Point2f> &obj_corners);

//Variables necesarias para el callback
bool drawing = false;
int xi = -1, yi = -1;

static void window_mouse_callback(int event, int x, int y, int flags, void* userdata);

Mat img_obj /*= imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE)*/, img_scene, frame, img_matches;
String window = "Object Tracker";

//Vectores de keypoints
vector<KeyPoint> kp_obj, kp_scene;

//Descriptores
Mat des_obj, des_scene;

//Matcher
FlannBasedMatcher matcher;
vector<DMatch> matches;

//Obtener esquinas del objeto
vector<Point2f> obj_corners(4), scene_corners(4);

bool rend = true, go = true; 

int main(int argc, char** argv){

    args(argc, argv);

    cout << "Camara " << cam << endl;
    cout << "Min matches " << min_match << endl;

    VideoCapture cap(cam);
    if(!cap.isOpened()) { cout << "No se pudo abrir la camara" << endl; readme(); }

    namedWindow(window, CV_WINDOW_AUTOSIZE);
    setMouseCallback(window, window_mouse_callback);

    while(go){
        if(rend) cap >> frame;
        img_scene = frame;
        if(bw) cvtColor(frame, img_scene, CV_BGR2GRAY);

        //Aplicar Surf a la escena
        if(img_obj.data && rend){
            surf(img_scene, kp_scene, des_scene);

            //Matchear los keypoints
            matcher.match(des_obj, des_scene, matches);
            
            //Buscar la distancia max y min entre matches
            double max_dist = 0, min_dist = 100;

            for(int i=0; i < matches.size(); i++){
                double dist = matches[i].distance;
                max_dist = dist > max_dist ? dist : max_dist;
                min_dist = dist < min_dist ? dist : min_dist;
            }

            //Obtener solo los matches buenos, aquellos que su dist sea < 3*min_dist
            vector<DMatch> good_matches;
            for(int i=0; i < matches.size(); i++){
                if(matches[i].distance < 2*min_dist) good_matches.push_back(matches[i]);
            }

            //Dibujamos los matches buenos
            if(rend)drawMatches(img_obj, kp_obj, img_scene, kp_scene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            //Dibujar el contorno solo si hay suficientes matches
            if(good_matches.size() > min_match){
                
                //Obtener los puntos de los matches por separado para findHomography y perspectiveTransform
                vector<Point2f> obj, scene;
                for(int i=0; i < good_matches.size(); i++){
                    obj.push_back(kp_obj[good_matches[i].queryIdx].pt);
                    scene.push_back(kp_scene[good_matches[i].trainIdx].pt);
                }

                Mat H = findHomography(obj, scene, CV_RANSAC);

                perspectiveTransform(obj_corners, scene_corners, H);

                //Dibujar las lineas del objeto en la escena
                if(rend){
                    line(img_matches, scene_corners[0] + Point2f(img_obj.cols, 0), scene_corners[1] + Point2f(img_obj.cols, 0), Scalar(0,255,0), 4);
                    line(img_matches, scene_corners[1] + Point2f(img_obj.cols, 0), scene_corners[2] + Point2f(img_obj.cols, 0), Scalar(0,255,0), 4);
                    line(img_matches, scene_corners[2] + Point2f(img_obj.cols, 0), scene_corners[3] + Point2f(img_obj.cols, 0), Scalar(0,255,0), 4);
                    line(img_matches, scene_corners[3] + Point2f(img_obj.cols, 0), scene_corners[0] + Point2f(img_obj.cols, 0), Scalar(0,255,0), 4);
                }
            }
        }
        //Mostrar el resultado
        if(!drawing) imshow(window, !img_matches.data ? img_scene : img_matches);
        switch(waitKey(1)){
            case 1048603:
                go = false;
                break;
            case 1048608:
                imwrite("captura.png", img_matches);
                break;
            case 1048690:
                rend = !rend;
                break;
        }
    }

    return 0;
}

void readme(){
    cout << "USO: ./surf_object_detector [-bw] [-c camera] [-m min_matches]" << endl;
    exit(-1);
}

void surf(Mat &img, vector<KeyPoint> &kp, Mat &des){
    //Detectar los puntos clave usando SURF Detectar
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);

    detector.detect(img, kp);
    
    //Calcular los descriptores
    SurfDescriptorExtractor extractor;

    extractor.compute(img, kp, des);
}

void surfObject(Mat &img_obj, vector<KeyPoint> &kp_obj, Mat &des_obj, vector<Point2f> &obj_corners){
    //Aplicar surf
    surf(img_obj, kp_obj, des_obj);

    //Calcular esquinas
    obj_corners[0] = Point2f(0,0); obj_corners[1] = Point2f(img_obj.cols, 0);
    obj_corners[2] = Point2f(img_obj.cols, img_obj.rows); obj_corners[3] = Point2f(0, img_obj.rows);
}

static void window_mouse_callback(int event, int x, int y, int flags, void* userdata){
    if(event == EVENT_LBUTTONDOWN){
        rend = false;
        drawing = true;
        xi = x - img_obj.cols;
        yi = y;
    } else if(event == EVENT_LBUTTONUP){
        drawing = false;
        if(xi == x && yi == y){ rend = true; return; }
        //Matrices necesarias
        Rect rect(Point(xi,yi), Point(x - img_obj.cols,y));
        Mat roi(img_scene, rect);
        img_obj = roi.clone();
        surfObject(img_obj, kp_obj, des_obj, obj_corners);
        rend = true;
    }else if(event == EVENT_MOUSEMOVE && drawing){
            Mat aux;
        if(!img_matches.data){
            aux = img_scene.clone();
            rectangle(aux, Rect(Point(xi,yi), Point(x -img_obj.cols,y)), Scalar(0,255,0), 3);
        }else{
            aux = img_matches.clone();
            rectangle(aux, Rect(Point(xi +img_obj.cols,yi), Point(x,y)), Scalar(0,255,0), 3);
        }
        imshow(window, aux);
    }
}

void args(int argc, char **argv){
    for(int i=1; i < argc; i++){
        if(bw_arg.compare(argv[i]) == 0){
            bw = true;
        }else if(min_match_arg.compare(argv[i]) == 0){
            i++; min_match = stoi(argv[i]);
        }else if(cam_arg.compare(argv[i]) == 0){
            i++; cam = stoi(argv[i]);
        }else if(help_arg.compare(argv[i]) == 0){
            man();
        }
    }
}

void man(){
    cout << endl << "Este programa hace uso del algoritmo de correspondencia discreta SURF." << endl << "Mientras el programa este en ejecucion selecciona una porcion de la ventana con el raton, para elegir donde aplicar el algoritmo." << endl << "Podras pausar la captura con r, y tomar una captura con la barra espaciadora" << endl;
    cout << endl << "USO:" << endl;
    cout << endl << "\t./surf_object_detector [-bw] [-c camera] [-m min_matches]" <<endl;
    cout << endl << "OPCIONES" << endl;
    cout << endl << "\t-c <camera>\tElige la camara a utilizar." << endl;
    cout << endl << "\t-m <min_matches>\tElige el numero de matches necesario para dibujar el objeto" << endl;
    cout << endl << "\t-bw\tEnciende el modo blanco y negro." << endl;
    cout << endl << "\t-h\tMuestra esta ayuda." << endl;
    exit(0);
}
