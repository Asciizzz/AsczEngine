#ifndef FPSHANDLE_H
#define FPSHANDLE_H

#include <iostream>
#include <chrono>
#include <thread>

using namespace std::chrono;

class FpsHandle {
public:
    FpsHandle() {};

    // Values
    int TARGET_FPS = 500;
    double TARGET_FTIME = 1000 / TARGET_FPS;
    double MAX_FTIME = 1000 / 5;

    // Frame handler
    double dTime = 0;
    double dTimeSec = 0;
    high_resolution_clock::time_point prevFTime = high_resolution_clock::now();

    // Other
    int fps = 0;
    int fpsAccumulate = 0;
    int fCount = 0;

    double startFrame();
    void endFrame();
};

#endif