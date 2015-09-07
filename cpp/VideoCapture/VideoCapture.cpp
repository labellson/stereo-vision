#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool test = false;

void args(int argc, char **argv){
    for(int i=1; i < argc; i++){
        if(((String)"-t").compare(argv[i]) == 0)
            test = true;
    }
}

int main (int argc, char **argv){
    //int camera;
    //cout << "Camara a usar: "; cin >> camera;
   
    args(argc, argv);

    VideoCapture cap(0);
    //cap.set(CV_CAP_PROP_FPS, 30);
    //cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    VideoCapture cap1(1);
    //cap1.set(CV_CAP_PROP_FPS, 30);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH,320);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    if(!cap.isOpened()){ cout << "La camara no se pudo abrir" << endl; return -1;}
    if(!cap1.isOpened()){ cout << "La camara no se pudo abrir" << endl; return -1;}
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
