In this repository are a bunch of programs written in C++ and developed with OpenCV library. For more information about compiling or usage see the [wiki](https://github.com/labellson/stereo-vision/wiki).
### List of Programs
* **VideoCapture:** Capture images with two cameras and make a performance test to obtain the framerate
* **calibrar:** Using a calibration pattern with this program you can calibrate one camera
* **calibrateStereo:** Using a calibration pattern with this program you can calibrate a stereoscopic system composed of two cameras
* **stereo_matchSGBM:** Obtains a disparity map of the scene using the SGBM (Semi Global Block Matching) algorithm
* **stereo_matchBM:** Obtains a disparity map of the scene using the BM (Block Matching) algorithm
* **stereo_matchBM_threads:** The same program than `stereo_matchBM`, but this program runs in parallel
* **SurfObjectDetector:** This program use the SURF algorithm to detect known objects in the scene

***

En este repositorio se encuentra una serie de programas escritos en C++ desarrollados con la libreria OpenCV. Para ver como compilarlos o utilizarlos mira en la [wiki](https://github.com/labellson/stereo-vision/wiki).
### Listado de Programas
* **VideoCapture:** Captura imagenes desde dos camaras y hace una prueba de rendimiento para obtener los FPS de la grabación
* **calibrar:** Utilizando un patrón de calibración con este programa calibraremos una unica cámara
* **calibrateStereo:** Utilizando un patrón de calibración calibraremos un sistema estereoscópico compuesto por dos cámaras
* **stereo_matchSGBM:** Obtiene un mapa de disparidad de la escena utilizando el algoritmo SGBM (Semi Global Block Matching)
* **stereo_matchBM:** Obtiene un mapa de disparidad de la escena utilizando el algoritmo BM (Block Matching)
* **stereo_matchBM_threads:** Igual que stereo_matchBM, pero el programa se ejecuta en paralelo
* **SurfObjectDetector:** Este programa hace uso del algoritmo SURF para detectar un objeto ya conocido
