# ShapeMask

A filter plugin for AviSynth that identifies bright areas such as projector
screens at conferences and creates a mask out of them. The use case for which
it was created is to deal with overly bright lectures, such as screen casts
or talks where the speaker or producer hasn't followed presentation 101;
*Use light text on a dark background!*

## Usage
The filter signature is as follows
```
ShapeMask ( int "thresh", float "minarea", bool "rectonly" )
```
Options:

* `thresh`: Luminance threshold (0-255)  
Specifies the lower inclusive limit for areas of the picture to consider for
mask inclusion (colored white). Anything below this value will be excluded
(colored black).   
Default: `127`

* `minarea`: Minimum area a shape must have to be included in the mask.   
This option is for filtering out noise, that is, very small shapes such as
dots and specks. If the value is below or equal to 1, then it is regarded as
a fraction of the clip's image size. If the value is larger than 1, then it is
regarded as pixel area. E.g. 0.05 means a shape must encompass at least 5% of
the image. 64 means 64 pixels, such as an 16x4 area (need not be rectangular).   
Default: `0.02`

* `rectonly`: Only consider shapes that are rectangular.   
That is have corners at about 90 degree angle. Particularly useful for screen
casts.   
Default: `false`

## Example
The following example creates a mask of a background projection, inverting the
blinding projection to produce a more ergonomic viewing while retaining high
contrast, so that what's being projected is still readable. 
```
mask = ShapeMask(thresh=160, minarea=0.005).blur(1.5)
Overlay(Invert, mask=mask)
```
Output: 

![Original image][orig_img] ![Processed image][processed_img]

*To the left is the original image and to the right the processed result using
the above example script.*

## Why
Created this because I watch a lot of recorded talks and webinars, but
my eyes can't stand looking into a light bulb for an hour straight.

## License
Same base license as AviSynth; GNU GPL v2 or later.  
The included OpenCV files (in the cv directory) are licensed under 3 clause BSD.

## Contribute
Fork it, hack away and send a pull request.

[orig_img]: http://griffeltavla.files.wordpress.com/2015/07/shapemask_example_orig.jpg
[processed_img]: http://griffeltavla.files.wordpress.com/2015/07/shapemask_example_processed.jpg
