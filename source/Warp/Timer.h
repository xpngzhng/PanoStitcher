#pragma once

#include <opencv2/core/core.hpp>

namespace ztool
{

class Timer
{
public:
    Timer(void) 
    {
        freq = cv::getTickFrequency(); 
        startTime = cv::getTickCount();
    };
    void start(void) 
    {
        startTime = cv::getTickCount();
    };
    void end(void)
    {
        endTime = cv::getTickCount();
    }
    double elapse(void) 
    {
        return double(endTime - startTime) / freq;
    };
private:
    double freq;
    long long int startTime, endTime;
};

class RepeatTimer
{
public:
    RepeatTimer(void) 
    {
        clear();
    };
    void clear(void) 
    {
        accTime = 0; 
        count = 0;
    }
    void start(void) 
    {
        timer.start();
    };
    void end(void) 
    {
        timer.end();
        accTime += timer.elapse(); 
        count++;
    };
    double getAccTime(void) 
    {
        return accTime;
    };
    int getCount(void) 
    {
        return count;
    };
    double getAvgTime(void) 
    {
        return count == 0 ? 0 : (accTime / count);
    }
private:    
    Timer timer;
    double accTime;
    int count;
};

class InterruptTimer
{
public:
    InterruptTimer(void)
    {
        accTime = 0;
    }
    void start(void)
    {
        timer.start();
    }
    void end(void)
    {
        timer.end();
        accTime += timer.elapse();
    }
    double elapse(void)
    {
        return accTime;
    }
    void clear(void)
    {
        accTime = 0;
    }
private:
    Timer timer;
    double accTime;
};

class RepeatInterruptTimer
{
public:
    RepeatInterruptTimer(void)
    {
        clear();
    }
    void clear(void)
    {
        accTime = 0;
        count = 0;
    }
    void start(void)
    {
        timer.start();
    }
    void end(void)
    {
        timer.end();
        accTime += timer.elapse();
    }
    void increaseCount(void)
    {
        count++;
    }
    double getAccTime(void) 
    {
        return accTime;
    };
    int getCount(void) 
    {
        return count;
    };
    double getAvgTime(void) 
    {
        return count == 0 ? 0 : (accTime / count);
    }
private:
    InterruptTimer timer;
    double accTime;
    int count;
};

}