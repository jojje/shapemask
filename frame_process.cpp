// ShapeMask: AviSynth filter for creating masks out of identified shapes.
// Copyright (C) 2015 Jonas Tingeborn
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
// USA.

#include <stdio.h>
#include <vector>
#include "cv/include/opencv2/imgproc.hpp"
#include "frame_process.hpp"

using namespace cv;
using namespace std;

void create_mask(Mat &gray, Mat &mask, int thresh, float minarea, bool rectonly) {
	int imgarea = gray.cols * gray.rows;
	double contarea;
	Mat binary;
	Scalar color = Scalar(255);                    // white fill color for the mask

	vector<vector<Point>> contours;                // holds all identified contours
	vector<vector<Point>> tmp_contours;            // subset of contours, used for drawing the filtered ones
	vector<Vec4i> hierarchy;
	vector<Point> approx;

	if (minarea <= 1) minarea = minarea * imgarea;

	threshold(gray, binary, thresh, 255, THRESH_BINARY);
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	if (contours.size() == 0) return;              // no contours found

	for (vector<Point> contour: contours) {
		if (rectonly) {
			approxPolyDP(contour, approx, 0.01 * arcLength(contour, true), true);
			if (approx.size() != 4) continue;
		}
		contarea = contourArea(contour);
		if (contarea < minarea) continue;

		tmp_contours.push_back(contour);
	}
	drawContours(mask, tmp_contours, -1, color, -1);
	tmp_contours.clear();
}

// Interface for Avisynth and other c-programs that have no OpenCV knowledge
unsigned char*
process_frame(unsigned char* pixels, int width, int height, int pitch, int colorspace,
              int threshold, float minarea, bool rectonly) {
	Mat src, gray;

	// Reduce image to one channel
	switch (colorspace) {
	case RGB24:
		src = Mat(height, width, CV_8UC3, pixels, pitch);
		cvtColor(src, gray, COLOR_RGB2GRAY);
		break;
	case RGB32:
		src = Mat(height, width, CV_8UC4, pixels, pitch);
		cvtColor(src, gray, COLOR_RGBA2GRAY);
		break;
	case YUV2:
		src = Mat(height, width, CV_8UC2, pixels, pitch);
		cvtColor(src, gray, COLOR_YUV2GRAY_YUY2);
		break;
	case YV12:
		src = Mat(height + (height / 2), width, CV_8UC1, pixels, pitch);
		cvtColor(src, gray, COLOR_YUV2GRAY_YV12);
		break;
	default:
		return 0;
	}

	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	create_mask(gray, mask, threshold, minarea, rectonly);

	int size = mask.cols * mask.rows;
	unsigned char* ret = new unsigned char[size];    // allocate new memory that doesn't vanish from the stack when returned
	memcpy(ret, mask.data, size);

	return ret;
}
