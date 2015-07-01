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

#include <Windows.h>
#include <stdio.h>
#include "shapemask.hpp"
#include "frame_process.hpp"

void raiseError(IScriptEnvironment* env, const char* msg);
PClip toGrayScale(IScriptEnvironment* env, PClip clip);
void copyRGB(const uchar* srcp, PVideoFrame &dst, int bpp);
void copyYUY2(const uchar* srcp, PVideoFrame &dst);
void copyPlanar(const uchar* srcp, PVideoFrame &dst, int bpp);

// ==========================================================================
// PUBLIC methods
// ==========================================================================

PVideoFrame __stdcall ShapeMask::GetFrame(int n, IScriptEnvironment* env) {
	int colorspace;

	if (vi.IsRGB24())      colorspace = RGB24;
	else if (vi.IsRGB32()) colorspace = RGB32;
	else if (vi.IsYUY2())  colorspace = YUV2;
	else if (vi.IsYV12())  colorspace = YV12;
	else raiseError(env, "Unsupported color space, must be one of RGB24, RGB32, YUV2 or YV12");

	PClip srcClip = toGrayScale(env, child);
	PVideoFrame src = srcClip->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);

	const uchar* srcp = src->GetReadPtr();
	const int src_pitch = src->GetPitch();
	const int bpp = vi.BitsPerPixel();

	uchar* retp;

	// No change to the source pixels in the process steps, so ok to cast to non-const
	// returns a 1 channel gray scale image which needs to be converted to whatever format the source clip is in.
	retp = process_frame((uchar*)srcp, vi.width, vi.height, src_pitch, colorspace, threshold, minarea, rectonly);
	if (retp == 0) raiseError(env, "Unsupported color space, must be one of RGB24, RGB32, YUV2 or YV12");

	if (vi.IsPlanar()) copyPlanar(retp, dst, bpp);
	else if (vi.IsYUY2()) copyYUY2(retp, dst);
	else copyRGB(retp, dst, bpp);

    delete retp;
	return dst;
}

// Constructor
ShapeMask::ShapeMask(PClip _child, int _thresh, float _minarea, bool _rectonly, IScriptEnvironment* env) :
GenericVideoFilter(_child),
threshold(_thresh),
minarea(_minarea),
rectonly(_rectonly) {
	if (threshold < 0 || threshold > 255) raiseError(env, "Threshold must be between 0-255");
}

ShapeMask::~ShapeMask() {}

AVSValue __cdecl Create_ShapeMask(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new ShapeMask(args[0].AsClip(),
		args[1].AsInt(127),         // threshold parameter
		args[2].AsFloat(0.02),      // minarea parameter
		args[3].AsBool(false),      // rectonly parameter
		env);
}


// ========================================================================
//  Helper functions
// ========================================================================

void raiseError(IScriptEnvironment* env, const char* msg) {
	char buff[1024];
	sprintf_s(buff, sizeof(buff), "[ShapeMask] %s", msg);
	env->ThrowError(buff);
}

PClip toGrayScale(IScriptEnvironment* env, PClip clip) {
	AVSValue args[1] = { clip };
	return env->Invoke("Grayscale", AVSValue(args, 1)).AsClip();
}

void copyRGB(const uchar* srcp, PVideoFrame &dst, int bpp) {
	const int bytes_per_pixel = bpp / 8;
	const int cols = dst->GetRowSize();
	const int rows = dst->GetHeight();
	const int dst_pitch = dst->GetPitch();
	const int src_pitch = cols / bytes_per_pixel;
	uchar* dstp = dst->GetWritePtr();
	int y, x;

	for (y = 0; y < rows; y++) {
		for (x = 0; x < cols; x++) {
			dstp[x] = srcp[x / bytes_per_pixel];   // copy the same mask pixel to all channels
		}
		srcp += src_pitch;
		dstp += dst_pitch;
	}
}

void copyYUY2(const uchar* srcp, PVideoFrame &dst) {
	const int cols = dst->GetRowSize();
	const int rows = dst->GetHeight();
	const int dst_pitch = dst->GetPitch();
	const int width = cols / 2;
	const int src_pitch = width;
	uchar* dstp = dst->GetWritePtr();
	int y, x;

	for (y = 0; y < rows; y++) {
		for (x = 0; x < width; x++) {
			dstp[x * 2] = srcp[x];                 // copy luma from mask
			dstp[x * 2 + 1] = 128;                 // set chroma to none
		}
		srcp += src_pitch;
		dstp += dst_pitch;
	}
}

void copyPlanar(const uchar* srcp, PVideoFrame &dst, int bpp) {
	const int bytes_per_pixel = bpp / 8;
	const int cols = dst->GetRowSize();
	const int rows = dst->GetHeight();
	const int dst_pitch = dst->GetPitch();
	const int src_pitch = cols / bytes_per_pixel;
	uchar* dstp = dst->GetWritePtr();
	int y, x;

	const int dst_pitchUV = dst->GetPitch(PLANAR_U);
	const int dst_widthUV = dst->GetRowSize(PLANAR_U);
	const int dst_heightUV = dst->GetHeight(PLANAR_U);

	for (y = 0; y < rows; y++) {                 // copy mask to Y plane
		for (x = 0; x < cols; x++) {
			dstp[x] = srcp[x / bytes_per_pixel];
		}
		srcp += src_pitch;
		dstp += dst_pitch;
	}

	dstp = dst->GetWritePtr(PLANAR_U);           // set U plane to no color

	for (y = 0; y < dst_heightUV; y++) {
		for (x = 0; x < dst_widthUV; x++) {
			dstp[x] = 128;                       // set chroma to none
		}
		dstp += dst_pitchUV;
	}

	// set V plane to no color
	dstp = dst->GetWritePtr(PLANAR_V);

	for (y = 0; y < dst_heightUV; y++) {
		for (x = 0; x < dst_widthUV; x++) {
			dstp[x] = 128;                       // set chroma to none
		}
		dstp += dst_pitchUV;
	}
}
