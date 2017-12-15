//============================================================================
// Name        : KinectTouch.cpp
// Author      : github.com/robbeofficial
// Version     : 0.something
// Description : recognizes touch points on arbitrary surfaces using kinect
// 				 and maps them to TUIO cursors
// 				 (turns any surface into a touchpad)
//============================================================================

/*
 * 1. point your kinect from a higher place down to your table
 * 2. start the program (keep your hands off the table for the beginning)
 * 3. use your table as a giant touchpad
 */

#include <iostream>
#include <vector>
#include <map>
using namespace std;

// openCV
#include <opencv/highgui.h>
#include <opencv/cv.h>
using namespace cv;

// openNI
#include <XnOpenNI.h>
#include <XnCppWrapper.h>
using namespace xn;
#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
		printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
		return rc;													\
	}

// TUIO

#include "TuioServer.h"
using namespace TUIO;

// TODO smoothing using kalman filter

//---------------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------------

// OpenNI
xn::Context xnContext;
xn::DepthGenerator xnDepthGenerator;
xn::ImageGenerator xnImgeGenertor;

bool mousePressed = false;

//---------------------------------------------------------------------------
// Functions
//---------------------------------------------------------------------------

int initOpenNI(const XnChar* fname) {
	XnStatus nRetVal = XN_STATUS_OK;
	ScriptNode scriptNode;

	// initialize context
	nRetVal = xnContext.InitFromXmlFile(fname, scriptNode);
	CHECK_RC(nRetVal, "InitFromXmlFile");

	// initialize depth generator
	nRetVal = xnContext.FindExistingNode(XN_NODE_TYPE_DEPTH, xnDepthGenerator);
	CHECK_RC(nRetVal, "FindExistingNode(XN_NODE_TYPE_DEPTH)");

	// initialize image generator
	nRetVal = xnContext.FindExistingNode(XN_NODE_TYPE_IMAGE, xnImgeGenertor);
	CHECK_RC(nRetVal, "FindExistingNode(XN_NODE_TYPE_IMAGE)");

	return 0;
}

void average(vector<Mat1s>& frames, Mat1s& mean) {
	Mat1d acc(mean.size());
	Mat1d frame(mean.size());

	for (unsigned int i=0; i<frames.size(); i++) {
		frames[i].convertTo(frame, CV_64FC1);
		acc = acc + frame;
	}

	acc = acc / frames.size();

	acc.convertTo(mean, CV_16SC1);
}

int main() {

	const unsigned int nBackgroundTrain = 30;
	const unsigned short touchDepthMin = 15;
	const unsigned short touchDepthMax = 25;
	const unsigned int touchMinArea = 50;

	const bool localClientMode = true; 					// connect to a local client

	const double debugFrameMaxDepth = 4000; // maximal distance (in millimeters) for 8 bit debug depth frame quantization
	const char* windowName = "Debug";
	const Scalar debugColor0(0,0,128);
	const Scalar debugColor1(255,0,0);
	const Scalar debugColor2(255,255,255);

	// int xMin = 110;
	// int xMax = 560;
	// int yMin = 120;
	// int yMax = 320;

	int xMin = 124;
	int xMax = 540;
	int yMin = 124;
	int yMax = 413;
	int slope = 23;

	Mat1s depth(480, 640); // 16 bit depth (in millimeters)
	Mat1b depth8(480, 640); // 8 bit depth
	Mat3b rgb(480, 640); // 8 bit depth

	Mat3b debug(480, 640); // debug visualization

	Mat1s foreground(640, 480);
	Mat1b foreground8(640, 480);

	Mat1b touch(640, 480); // touch mask

	Mat1s background(480, 640);
	vector<Mat1s> buffer(nBackgroundTrain);

	initOpenNI("../niConfig.xml");

	// create some sliders
	namedWindow(windowName);
	createTrackbar("xMin", windowName, &xMin, 640);
	createTrackbar("xMax", windowName, &xMax, 640);
	createTrackbar("yMin", windowName, &yMin, 480);
	createTrackbar("yMax", windowName, &yMax, 480);
	createTrackbar("Slope", windowName, &slope, 320);

	// create background model (average depth)
	for (unsigned int i=0; i<nBackgroundTrain; i++) {
		xnContext.WaitAndUpdateAll();
		depth.data = (uchar*) xnDepthGenerator.GetDepthMap();
		buffer[i] = depth;
	}
	average(buffer, background);

	// Detected touched coordinates
  float minx = 1;
	float maxx = 0;
	float miny = 1;
	float maxy = 0;

	while ( (char) waitKey(1) != (char) 27 ) {
		// read available data
		xnContext.WaitAndUpdateAll();

		// update 16 bit depth matrix
		depth.data = (uchar*) xnDepthGenerator.GetDepthMap();
		//xnImgeGenertor.GetGrayscale8ImageMap();

		// update rgb image
		//rgb.data = (uchar*) xnImgeGenertor.GetRGB24ImageMap(); // segmentation fault here
		//cvtColor(rgb, rgb, CV_RGB2BGR);

		// extract foreground by simple subtraction of very basic background model
		foreground = background - depth;

		// find touch mask by thresholding (points that are close to background = touch points)
		touch = (foreground > touchDepthMin) & (foreground < touchDepthMax);

		// extract ROI
		Rect roi(xMin, yMin, xMax - xMin, yMax - yMin);

    Point rightttop(xMax-slope,yMin);
    Point rightbot(xMax,yMax);
    Point leftbot(xMin,yMax);
    Point lefttop(xMin+slope,yMin);
    vector< vector<Point> >  trapezoid;
    trapezoid.push_back(vector<Point>());
    trapezoid[0].push_back(rightttop);
    trapezoid[0].push_back(rightbot);
    trapezoid[0].push_back(leftbot);
    trapezoid[0].push_back(lefttop);

    Mat touchRoi = touch(roi);

		// find touch points
		vector< vector<Point2i> > contours;
		vector<Point2f> touchPoints;
		findContours(touchRoi, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point2i(xMin, yMin));
    float fslope = slope; // slope as float for calc
		float slopePerPix = fslope / (yMax-yMin);
    for (unsigned int i=0; i<contours.size(); i++) {
			Mat contourMat(contours[i]);
			// find touch points by area thresholding
			if ( contourArea(contourMat) > touchMinArea ) {
				// Touch is inside rectangle ROI
				Scalar center = mean(contourMat);
        float touchx = center[0] - xMin;
        float touchy = center[1] - yMin;
        float slopedAtCurrY = ((yMax-yMin) - touchy) * slopePerPix;

				if (touchx < (xMax-xMin-slopedAtCurrY) && touchx > slopedAtCurrY) {
					// Touch is inside trapezoid
					Point2f touchPoint(touchx, touchy);
					touchPoints.push_back(touchPoint);
				}
			}
		}

		for (unsigned int i=0; i<touchPoints.size(); i++) { // touch points
      float slopedAtCurrHeight = ((yMax-yMin) - touchPoints[i].y) * slopePerPix;

			float cursorX = (touchPoints[i].x - slopedAtCurrHeight) / (xMax - xMin - 2 * slopedAtCurrHeight);
			float cursorY = 1 - (touchPoints[i].y) / (yMax - yMin);
      printf("cursorX: %f, cursorY: %f\n", cursorX, cursorY);

      // record touched area
			if (cursorX < minx) minx = cursorX;
			if (cursorX > maxx) maxx = cursorX;
			if (cursorY < miny) miny = cursorY;
			if (cursorY > maxy) maxy = cursorY;
		}

		// draw debug frame
		depth.convertTo(depth8, CV_8U, 255 / debugFrameMaxDepth); // render depth to debug frame
		cvtColor(depth8, debug, CV_GRAY2BGR);
		debug.setTo(debugColor0, touch);  // touch mask
		polylines(debug, trapezoid, true, debugColor1, 3); // surface boundaries

    for (unsigned int i=0; i<touchPoints.size(); i++) { // touch points
			circle(debug, Point2i(touchPoints[i].x + xMin, touchPoints[i].y + yMin), 5, debugColor2, CV_FILLED);
		}

		// render debug frame (with sliders)
		imshow(windowName, debug);
		// imshow("image", rgb);
	}

	printf("Min X is %f\n", minx);
	printf("Max X is %f\n", maxx);
	printf("Min Y is %f\n", miny);
  printf("Max Y is %f\n", maxy);
	return 0;
}
