#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace std;
using namespace cv;

bool test = false;

String size_arg = "-s", help_arg = "-h";

char *cam0_str, *cam1_str;
int w = 320, h = 240;

void man(){
    cout << endl << "Este programa inicia inicia una grabacion utilizando dos camaras" <<endl; 
    cout << endl << "USO:" <<endl;
    cout << endl << "\t./VideoCapture <camara0> <camara1> [-s <width> <height>] [-t]" << endl;
    cout << endl << "OPCIONES:" << endl;
    cout << endl << "\t-s <width> <height>\tSirve para introducir la resolucion de la captura" << endl;
    cout << endl << "\t-t\tInicia el modo test del programa. Se ejecutaran 100 ciclos de programa y se calculara el promedio de tiempo para cada ciclo" <<endl;
    exit(0);
}

void args(int argc, char **argv){
    for(int i=1; i < argc; i++){
        if(argv[i][0] != '-'){
            if(!cam0_str) cam0_str = argv[i];
            else cam1_str = argv[i];
        }else if(((String)"-t").compare(argv[i]) == 0){
            test = true;
        }else if(size_arg.compare(argv[i]) == 0){
            i++; w = stoi(argv[i]);
            i++; h = stoi(argv[i]);
        }else if(help_arg.compare(argv[i]) == 0){
            man();
        }
    }
}

void readme(){
    cout << endl << "USO: ./VideoCapture <camara0> <camara1> [-s <width> <height>] [-t]" << endl << "For more help type -h" << endl;
}


int main (int argc, char **argv){
    //int camera;
    //cout << "Camara a usar: "; cin >> camera;
   
    args(argc, argv);

    int cam0 = !cam0_str ? 0 : stoi(cam0_str), cam1 = !cam1_str ? 1 : stoi(cam1_str);

    VideoCapture cap(cam0);
    //cap.set(CV_CAP_PROP_FPS, 30);
    //cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    cap.set(CV_CAP_PROP_FRAME_WIDTH,w);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,h);
    VideoCapture cap1(cam1);
    //cap1.set(CV_CAP_PROP_FPS, 30);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH,w);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT,h);
    if(!cap.isOpened()){ cout << "La camara no se pudo abrir" << endl; readme(); return -1;}
    if(!cap1.isOpened()){ cout << "La camara no se pudo abrir" << endl; readme(); return -1;}
    Mat frame, frame1;
    namedWindow("Camara", CV_WINDOW_AUTOSIZE);
    vector<double> tiempos;
    bool first=true;
    while(1){
        double t = getTickCount();
        cap >> frame;
        cap1 >> frame1;
        imshow("Camara", frame);
        imshow("Camara1", frame1);
        if(waitKey(30) >= 0) break;
        double t_f = ((double)getTickCount() - t)/getTickFrequency(); 
        cout << "Tiempo " << t_f << "s" << endl;
        if(test && !first){
            tiempos.push_back(t_f);
            if(tiempos.size() >= 100) break;
        }else
            first = false;
    }
    if(test){
        imwrite("cap.png", frame);
        imwrite("cap1.png", frame1);
        double sum=0;
        int i;
        for(i=0; i<tiempos.size(); i++)
            sum+= tiempos[i];
        cout << "Media " << sum/i << "s" << endl;
    }
    return 0;
}
