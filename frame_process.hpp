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

#ifndef __FRAME_PROCESS_H
#define __FRAME_PROCESS_H

enum COLOR_SPACE { RGB24, RGB32, YUV2, YV12 };

/**
 * Performs the actual image processing of the plugin using OpenCV
 * pixels:     source pixels in row[array] order of tuples symbolizing colors
 * width:      image width (number of columns/bytes per color)
 * height:     image height (number of rows)
 * pitch:      width + some offset the image producing library adds for its deemed efficiency.
 * colorspace: bits per pixel
 * threshold:  luminence equal or above which pixels will be retained (white) in the mask and below will be masked out.
 * minarea:    The minimal area an identified shape must have to be regarded as a valid shape of interest, and be included in the visible mask (painted white).
 * rectonly:   Whether or not to only consider shapes that are rectangular (have corners at 90 deg angle, +- 1 degree)
 */
unsigned char* 
process_frame(unsigned char* pixels, int width, int height, int pitch, int colorspace,
	          int threshold, float minarea, bool rectonly);

#endif