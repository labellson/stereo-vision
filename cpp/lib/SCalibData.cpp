#include "SCalibData.h"

SCalibData::SCalibData(){
}

SCalibData::SCalibData(Mat CM1, Mat CM2, Mat D1, Mat D2, Mat R, Mat T, Mat E, Mat F, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, Rect roi1, Rect roi2){
    CM.push_back(CM1);
    CM.push_back(CM2);
    D.push_back(D1);
    D.push_back(D2);
    this->R = R;
    this->T = T;
    this->E = E;
    this->F = F;
    r.push_back(R1);
    r.push_back(R2);
    P.push_back(P1);
    P.push_back(P2);
    this->Q = Q;
    /*vector<Mat> map1;
    vector<Mat> map2;
    map1.push_back(map1x);
    map1.push_back(map1y);
    map2.push_back(map2x);
    map2.push_back(map2y);
    map.push_back(map1);
    map.push_back(map2);*/
    roi.push_back(roi1);
    roi.push_back(roi2);
}

void SCalibData::fillVector(vector<Mat> &v, FileNode &n){
    FileNodeIterator it = n.begin(), it_end = n.end();
    for(; it != it_end; it++){
        Mat aux;
        *it >> aux;
        v.push_back(aux);
    }
}

void SCalibData::write(FileStorage& fs) const{
    fs << "CM" << "{" << "l" << CM[0] << "r" << CM[1] << "}";
    fs << "D" << "{" << "l" << D[0] << "r" << D[1] << "}";
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs << "r" << "{" << "l" << r[0] << "r" << r[1] << "}";
    fs << "P" << "{" << "l" << P[0] << "r"  << P[1] << "}";
    fs << "Q" << Q;
    //fs << "map1" << "{" << "x" << map[0][0] << "y" << map[0][1] << "}";
    //fs << "map2" << "{" << "x" << map[1][0] << "y" << map[1][1] << "}";
    fs << "roi" << "{" << "l" << roi[0] << "r" << roi[1] << "}";
}

void SCalibData::read(const FileStorage& fs){
    FileNode n = fs["CM"];
    fillVector(CM, n);
    n = fs["D"];
    fillVector(D, n);
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    n = fs["r"];
    fillVector(r, n);
    n = fs["P"];
    fillVector(P, n);
    fs["Q"] >> Q;
    n = fs["roi"];
    Rect aux;
    n["l"] >> aux;
    roi.push_back(aux);
    n["r"] >> aux;
    roi.push_back(aux);
    /*n = fs["map1"];
    vector<Mat> map1;
    fillVector(map1, n);
    n = fs["map2"];
    vector<Mat> map2;
    fillVector(map2, n);
    map.push_back(map1);
    map.push_back(map2);*/
}

void write(FileStorage& fs, const std::string&, const SCalibData& x){
    x.write(fs);
}

void read(const FileStorage& fs, SCalibData& x){
    x.read(fs);
}
