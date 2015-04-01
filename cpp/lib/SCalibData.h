#ifndef __SCALIBDATA_H__
#define __SCALIBDATA_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/** Contiene los datos de un par estereo, ademas es compatible con FileStorage **/
class SCalibData{
    public:

        SCalibData();

        //SCalibData(Mat[] pCM, Mat[] pD, Mat pR, Mat pT, Mat pE, Mat pF, Mat[] pr, Mat[] pP, Mat pQ, Mat[][] pmap) : CM(pCM), D(pD), R(pR), T(pT), E(pE), F(pF), r(pr), P(pP), Q(pQ), map(pmap) {} 

        SCalibData(Mat CM1, Mat CM2, Mat D1, Mat D2, Mat R, Mat T, Mat E, Mat F, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, Mat map1x, Mat map1y, Mat map2x, Mat map2y);

        //Convertir a la clase en serializable
        void write(FileStorage& fs) const;

        void read(const FileStorage& fs);

        void fillVector(vector<Mat>& v, FileNode& n);

        vector<Mat>CM; //Matrices de calibracion de camaras
        vector<Mat> D; //Coeficientes de distorsion de camaras
        Mat R; //Matriz rotacion del conjunto binocular
        Mat T; //Matriz de Traslacion del conjunto binocular
        Mat E; //Matriz Esencial
        Mat F; //Matriz Fundamental
        vector<Mat> r; //Matrices de rotacion de las camaras
        vector<Mat> P; //Matrices de proyeccion de las camaras
        Mat Q; //Matriz disparidad a profundidad
        vector<vector <Mat> > map; //Mapeos para rectificar con remap. Primera dimension indica la camara. Segunda indica coordenadas x, y
};

void write(FileStorage& fs, const std::string&, const SCalibData& x);
void read(const FileStorage& fs, SCalibData& x);
#endif
