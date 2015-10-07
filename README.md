En este repositorio se encuentra una serie de programas escritos en C++ desarrollados con la libreria OpenCV. Para ver como utilizarlos ver la wiki.
### Listado de Programas
* **VideoCapture:** Captura imagenes desde dos camaras y hace una prueba de rendimiento para obtener los FPS de la grabación
* **calibrar:** Utilizando un patrón de calibración con este programa calibraremos una unica cámara
* **calibrateStereo:** Utilizando un patrón de calibración calibraremos un sistema estereoscópico compuesto por dos cámaras
* **stereo_matchSGBM:** Obtiene un mapa de disparidad de la escena utilizando el algoritmo SGBM (Semi Global Block Matching)
* **stereo_matchBM:** Obtiene un mapa de disparidad de la escena utilizando el algoritmo BM (Block Matching)
* **stereo_matchBM_threads:** Igual que stereo_matchBM, pero el programa se ejecuta en paralelo
* **SurfObjectDetector:** Este programa hace uso del algoritmo SURF para detectar un objeto ya conocido
