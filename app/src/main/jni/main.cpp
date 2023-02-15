#include <jni.h>
//#include "kr_ac_smu_firstopencv_example1_MainActivity.h"
#include <android/log.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include "DataExporter.hpp"

using namespace cv;

extern "C"{
    JNIEXPORT void JNICALL
    Java_kr_ac_smu_secondopencv_1example2_MainActivity_ConvertRGBtoGray(
            JNIEnv *env,
            jobject  instance,
            jlong matAddrInput,
            jlong matAddrResult){
        Mat &matInput = *(Mat *)matAddrInput;
        Mat &matResult = *(Mat *)matAddrResult;

        cvtColor(matInput, matResult, COLOR_RGBA2GRAY);
      }

    static unsigned long long GetCurrentTimeInNanoseconds()
    {
        timespec t;

        clock_gettime(CLOCK_MONOTONIC, &t);
        return (unsigned long long)(t.tv_sec*1000000000ULL) + (unsigned long long)t.tv_nsec;
    }

    void
    RectifyImage(Mat &matInput, Mat &matMtx, Mat &matDist, Mat &matResult) {
        Mat matNewCam, oaMap1, oaMap2;
        Rect roi;
        Size sz = matInput.size();

        matNewCam = getOptimalNewCameraMatrix(matMtx, matDist, sz, 0, sz, &roi);
        initUndistortRectifyMap(matMtx, matDist, cv::Mat(), matNewCam, sz, 5, oaMap1, oaMap2);
        remap(matInput, matResult, oaMap1, oaMap2, INTER_LINEAR);
    }

    void
    ResizeImage(Mat &matInput1, Mat &matInput2) {
        //Mat *pInput1 = new Mat(), *pInput2 = new Mat();

        resize(matInput1, matInput1, Size(480, 640), 0, 0, cv::INTER_AREA);
        resize(matInput2, matInput2, Size(480, 640), 0, 0, cv::INTER_AREA);

        /*Mat rgbFrame(960, 1280, CV_8UC3);
        cvtColor(matInput2, rgbFrame, CV_BGR2RGB);

        // ...now let it convert it to RGBA
        Mat newSrc = Mat(rgbFrame.rows, rgbFrame.cols, CV_8UC4);
        int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
        mixChannels(&rgbFrame, 2, &newSrc, 1, from_to, 4);

        matInput2 = newSrc;*/
        //matInput1 = *pInput1;
        //matInput2 = *pInput2;
    }

    Mat*
    CropImage(Mat matInput, Mat matReference, unsigned long long &llTime) {
        cv::Ptr<SIFT> f2d = SIFT::create(); //SIFT::create();

        std::vector<KeyPoint> kp1, kp2; // = new std::vector<KeyPoint>();
        cv::Mat des1, des2;

        f2d->detectAndCompute(matReference, Mat(), kp1, des1);
        f2d->detectAndCompute(matInput, Mat(), kp2, des2);
        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Finding)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - llTime);

        llTime = GetCurrentTimeInNanoseconds();

        //cv::flann::IndexParams indexParams;
        const static auto indexParams = new cv::flann::IndexParams();
        indexParams->setAlgorithm(cvflann::FLANN_INDEX_KDTREE);
        indexParams->setInt("trees", 5);

        //cv::flann::SearchParams searchParams;
        const static auto searchParams = new cv::flann::SearchParams();
        searchParams->setInt("checks", 50);

        std::vector<std::vector<cv::DMatch>> matches;
        const static auto flann = cv::FlannBasedMatcher(indexParams, searchParams);
        //flann.knnMatch(des1, des2, matches, 2);
        flann.knnMatch(des1, des2, matches, 2);

        int lmq = 0, lmt = 0, rmq = 0, rmt = 0, tmq = 0, tmt = 0, bmq = 0, bmt = 0;
        float lmqp = 1000.0, lmtp = 1000.0, rmqp = 0.0, rmtp = 0.0;
        float tmqp = 1000.0, tmtp = 1000.0, bmqp = 0.0, bmtp = 0.0;

        for(int i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < 0.6 * matches[i][1].distance)
            {
                if (kp1.at(matches[i][0].queryIdx).pt.x < lmqp) {
                    lmqp = kp1.at(matches[i][0].queryIdx).pt.x;
                    lmq = matches[i][0].queryIdx;
                }

                if (kp1.at(matches[i][0].queryIdx).pt.x > rmqp) {
                    rmqp = kp1.at(matches[i][0].queryIdx).pt.x;
                    rmq = matches[i][0].queryIdx;
                }

                if (kp1.at(matches[i][0].queryIdx).pt.y < tmqp) {
                    tmqp = kp1.at(matches[i][0].queryIdx).pt.y;
                    tmq = matches[i][0].queryIdx;
                }

                if (kp1.at(matches[i][0].queryIdx).pt.y > bmqp) {
                    bmqp = kp1.at(matches[i][0].queryIdx).pt.y;
                    bmq = matches[i][0].queryIdx;
                }

                if (kp2.at(matches[i][0].trainIdx).pt.x < lmtp) {
                    lmtp = kp2.at(matches[i][0].trainIdx).pt.x;
                    lmt = matches[i][0].trainIdx;
                }

                if (kp2.at(matches[i][0].trainIdx).pt.x > rmtp) {
                    rmtp = kp2.at(matches[i][0].trainIdx).pt.x;
                    rmt = matches[i][0].trainIdx;
                }

                if (kp2.at(matches[i][0].trainIdx).pt.y < tmtp) {
                    tmtp = kp2.at(matches[i][0].trainIdx).pt.y;
                    tmt = matches[i][0].trainIdx;
                }

                if (kp2.at(matches[i][0].trainIdx).pt.y > bmtp) {
                    bmtp = kp2.at(matches[i][0].trainIdx).pt.y;
                    bmt = matches[i][0].trainIdx;
                }

                i++;
            }
        }

        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Matching)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - llTime);
        llTime = GetCurrentTimeInNanoseconds();

        float ratiow = -1; //(kp1.at(rmq).pt.x - kp1.at(lmq).pt.x) / (kp2.at(rmt).pt.x - kp2.at(lmt).pt.x);
        float ratioh = -1; //(kp1.at(bmq).pt.y - kp1.at(tmq).pt.y) / (kp2.at(bmt).pt.y - kp2.at(tmt).pt.y);

        if (rmtp - lmtp > 0.0f && rmqp - lmqp > 0.0f) ratiow = (kp1.at(rmq).pt.x - kp1.at(lmq).pt.x) / (kp2.at(rmt).pt.x - kp2.at(lmt).pt.x);
        if (bmtp - tmtp > 0.0f && bmqp - tmqp > 0.0f) ratioh = (kp1.at(bmq).pt.y - kp1.at(tmq).pt.y) / (kp2.at(bmt).pt.y - kp2.at(tmt).pt.y);

        if (ratiow == -1 || ratioh == -1) return NULL;

        Mat matOutput;
        resize(matInput, matOutput, Size(ratiow * matInput.size().width, ratioh * matInput.size().height), 0, 0, INTER_LINEAR_EXACT);

        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Resizing)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - llTime);
        llTime = GetCurrentTimeInNanoseconds();

        float x = 0, y = 0;
        int mh = round(kp2.at(tmt).pt.y * ratioh - kp1.at(tmq).pt.y);
        int mw = round(kp2.at(lmt).pt.x * ratiow - kp1.at(lmq).pt.x) + 20;

        Rect r(x + mw, y + mh, matReference.cols, matReference.rows);// x + mw + matReference.cols, y + mh + matReference.rows);
        Rect bounds(0, 0, matOutput.cols, matOutput.rows);

        Mat imgL = matReference;
        Mat imgR = matOutput(r & bounds);

        if (imgL.size() != imgR.size())
        {
            int rowPadding = matReference.rows - imgR.rows;
            int colPadding = matReference.cols - imgR.cols;

            cv::copyMakeBorder(imgR, imgR, 0, rowPadding, 0, colPadding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        }

        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Cropping)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - llTime);
        llTime = GetCurrentTimeInNanoseconds();
#if 0
        Mat *disparity2 = new Mat(480, 640, CV_8UC4);
        *disparity2 = imgR;
#else
        Mat gray1, gray2;

        cvtColor(imgL, gray1, COLOR_BGR2GRAY);
        cvtColor(imgR, gray2, COLOR_BGR2GRAY);

        int window_size = 3, min_disp = -16, num_disp = -min_disp;
        Ptr<StereoSGBM> stereo1 = StereoSGBM::create(
                min_disp, num_disp, 25,
                pow(8 * 3 * window_size, 2),
                pow(32 * 3 * window_size,2),
                1,63,15,0,2,StereoSGBM::MODE_SGBM_3WAY);

        Ptr<StereoMatcher> stereo2 = cv::ximgproc::createRightMatcher(stereo1);

        double lmbda = 80000;
        float sigma = 1.2;
        float visual_multiplier = 1.0;
        Mat displ, dispr, *disparity2 = new Mat(480, 640, CV_8UC4);
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo1);
        wls_filter->setLambda(lmbda);
        wls_filter->setSigmaColor(sigma);

        /*if (gray1.size() != gray2.size() || gray1.type() != gray2.type() || gray1.depth() != CV_8U)
        {
            int a = 0;
        }*/
        stereo1->compute(gray1, gray2, displ);
        stereo2->compute(gray2, gray1, dispr);
        //displ = np.int16(displ);
        //dispr = np.int16(dispr);

        wls_filter->filter(displ, gray1, *disparity2, dispr);
        cv::normalize(*disparity2, *disparity2, 255, 0, NORM_MINMAX);
        //filteredImg = np.uint8(filteredImg);
        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Calculating the disparity map)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - llTime);
        llTime = GetCurrentTimeInNanoseconds();
#endif
         return disparity2;
    }

    void
    ReprojectImage(Mat matDisparity, Mat matReference, Mat matQ, Mat &matPoints, Mat &matColors) {
        //Mat points(matDisparity.size(),CV_32FC3), colors;
        //Mat *pPoints = new Mat(matDisparity.size(),CV_32FC3), *pColors = new Mat();

        reprojectImageTo3D(matDisparity, matPoints, matQ, false, CV_32F);
        cvtColor(matReference, matColors, COLOR_BGR2RGB);

        //DataExporter data(points, colors, "/data/local/tmp/out.ply", FileFormat::PLY_ASCII);
        //data.exportToFile();
    }

    JNIEXPORT void JNICALL
    Java_kr_ac_smu_secondopencv_1example2_MainActivity_GetLeftImage(JNIEnv *env, jobject thiz, jlong matAddrInput, jlong matAddrMtx, jlong matAddrDist, jlong matAddrResult) {
        Mat &matInput = *(Mat *)matAddrInput;
        Mat &matMtx = *(Mat *)matAddrMtx;
        Mat &matDist = *(Mat *)matAddrDist;
        Mat &matResult = *(Mat *)matAddrResult;

        RectifyImage(matInput, matMtx, matDist, matResult);
    }

    JNIEXPORT void JNICALL
    Java_kr_ac_smu_secondopencv_1example2_MainActivity_GetRightImage(JNIEnv *env, jobject thiz, jlong matAddrInput, jlong matAddrReference, jlong matAddrMtx, jlong matAddrDist, jlong matAddrQ, jlong matAddrResult, jlong matAddrPoints, jlong matAddrColors) {
        Mat &matInput = *(Mat *) matAddrInput;
        Mat &matReference = *(Mat *) matAddrReference;
        Mat &matMtx = *(Mat *) matAddrMtx;
        Mat &matDist = *(Mat *) matAddrDist;
        Mat &matQ = *(Mat *) matAddrQ;
        Mat &matResult = *(Mat *) matAddrResult;
        Mat &matPoints = *(Mat *) matAddrPoints;
        Mat &matColors = *(Mat *) matAddrColors;

        unsigned long long sTime = GetCurrentTimeInNanoseconds();
        RectifyImage(matInput, matMtx, matDist, matResult);
        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Rectifying)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - sTime);

        sTime = GetCurrentTimeInNanoseconds();
        ResizeImage(matReference, matResult);
        //__android_log_print(ANDROID_LOG_DEBUG,
        //                    "SecondOpenCV_Example2_main.cpp, Elapsed time(Resizing)", "%llu nsec",
        //                    GetCurrentTimeInNanoseconds() - sTime);

        //sTime = GetCurrentTimeInNanoseconds();
        Mat *pCroppedImg = CropImage(matResult, matReference, sTime);
        __android_log_print(ANDROID_LOG_DEBUG,
                            "SecondOpenCV_Example2_main.cpp, Elapsed time(Cropping)", "%llu nsec",
                            GetCurrentTimeInNanoseconds() - sTime);

        if (!pCroppedImg) return;

        //sTime = GetCurrentTimeInNanoseconds();
        //ReprojectImage(*pCroppedImg, matReference, matQ, matPoints, matColors);
        //__android_log_print(ANDROID_LOG_DEBUG,
        //                    "SecondOpenCV_Example2_main.cpp, Elapsed time(Reprojecting)", "%llu nsec",
        //                    GetCurrentTimeInNanoseconds() - sTime);

        matResult = *pCroppedImg; //matInput; //*pResizedImg;
    }
}
