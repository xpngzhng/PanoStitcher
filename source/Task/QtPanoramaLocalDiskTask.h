#pragma once

#include "PanoramaTask.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QProgressDialog>
#include <thread>

class QtCPUPanoramaLocalDiskTask : public CPUPanoramaLocalDiskTask
{
public:
    ~QtCPUPanoramaLocalDiskTask() {}
    bool run(QWidget* obj)
    {
        if (!start())
            return false;

        QProgressDialog* progressDialog = new QProgressDialog(obj);
        progressDialog->setMinimumSize(400, 50);
        progressDialog->resize(QSize(400, 50));
        progressDialog->setModal(true);
        progressDialog->setWindowModality(Qt::WindowModal);
        progressDialog->setWindowTitle(QObject::tr("Generating Panorama Video ......"));
        progressDialog->setRange(0, 100);
        progressDialog->setStyleSheet("background:rgb(88,88,88)");
        progressDialog->setValue(0);
        progressDialog->show();
        QCoreApplication::processEvents();

        int progress;
        while (true)
        {
            progress = getProgress();
            progressDialog->setValue(progress);
            if (progress >= 100)
                break;
            if (progressDialog->wasCanceled())
                cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        waitForCompletion();

        return true;
    }
};

class QtCudaPanoramaLocalDiskTask : public CudaPanoramaLocalDiskTask
{
public:
    ~QtCudaPanoramaLocalDiskTask() {}
    bool run(QWidget* obj)
    {
        if (!start())
            return false;

        QProgressDialog* progressDialog = new QProgressDialog(obj);
        progressDialog->setMinimumSize(400, 50);
        progressDialog->resize(QSize(400, 50));
        progressDialog->setModal(true);
        progressDialog->setWindowModality(Qt::WindowModal);
        progressDialog->setWindowTitle(QObject::tr("Generating Panorama Video ......"));
        progressDialog->setRange(0, 100);
        progressDialog->setStyleSheet("background:rgb(88,88,88)");
        progressDialog->setValue(0);
        progressDialog->show();
        QCoreApplication::processEvents();

        int progress;
        while (true)
        {
            progress = getProgress();
            progressDialog->setValue(progress);
            if (progress >= 100)
                break;
            if (progressDialog->wasCanceled())
                cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        waitForCompletion();

        return true;
    }
};