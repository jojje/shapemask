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

#ifndef __SHAPEMASK_H
#define __SHAPEMASK_H

#ifdef __GNUC__
#define _ASSERT(x)
#endif

#include "avisynth/avisynth.h"

#define VERSION "1.0.2"

class ShapeMask : public GenericVideoFilter {
	int   threshold;          // The luma threshold (0-255) above which pixels must be to be considered members of a shape candidate.
	float minarea;            // Smallest area that a feature in the picture has to have in order to be considered a viable shape. If <= 1 then it's taken as a percentage of the entire image area, else as the area in pixels.
	bool  rectonly;           // Whether or not to only consider rectangular shapes to include in the shape mask (true|false).
public:
	ShapeMask(PClip _child, int thresh, float minarea, bool rectonly, IScriptEnvironment* env);
	~ShapeMask();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};

AVSValue __cdecl Create_ShapeMask(AVSValue args, void* user_data, IScriptEnvironment* env);

// shapemask(clip, threshold = 127, minarea = 0.02, rectonly = true)
extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env) {
	env->AddFunction("ShapeMask", "c[THRESH]i[MINAREA]f[RECTONLY]b", Create_ShapeMask, 0);
	return "'ShapeMask' plugin v" VERSION ", author: tinjon[at]gmail.com";
}


// for 64bit AVSP
const AVS_Linkage* AVS_linkage = 0;  // new requirement for avisynth+ apparently see: https://github.com/AviSynth/AviSynthPlus/blob/v3.7.2/plugins/ConvertStacked/ConvertStacked.cpp#L360
extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("ShapeMask", "c[THRESH]i[MINAREA]f[RECTONLY]b", Create_ShapeMask, 0);
	return "'ShapeMask' plugin v" VERSION ", author: tinjon[at]gmail.com";
}

#endif