Framephoto
==========

A simple tool to auto-crop/fit and fill images of arbitrary size to a fixed 
resolution, e.g. for use in a digital photo frame.

If the aspect ratio of source and target are similar, will use 
[Katna](https://github.com/keplerlab/Katna) to find a suitable crop and then
resize to the target resolution.

If the source and target aspect ratios are too dissimilar (e.g. portrait
photo on a landscape frame), it will fit the image to the target resolution
and fill the open space using either an inpainting algorithm or a solid color.


Installation
------------

Requires Python 3.6. It is recommended to create a virtual environment:

    pip install virtualenv
    virtualenv -p python3 venv

Then install the requirements:

    venv/bin/pip install -r requirements.txt

It seems the current version of Katna may fail to install if numpy is not
already installed, even though it is listed in the requirements. If that
happens, first install numpy separately:

    venv/bin/pip install numpy

and then try the previous step again.


Usage
-----

Everything should be self-explanatory from the help, which you can access with

    venv/bin/python framephoto.py --help

Helpfully, we included a bash script to act as a shorthand for running
framephoto.py in the virtualenv, so if you have set up the virtualenv 
in `venv/` you can also use:

    ./framephoto --help


There are basically two modes of operation:

1. Specifying individual images to process
2. Specifying a source folder and processing it recursively

Option 1 is straightforward; all specified images are processed and put in the
destination folder, keeping the original filename.

Option 2 is activated using the `-r` or `--recurse` switch. In that case, the
source path should be a folder, and its directory structure is replicated in
the destination folder. Already-processed images are not touched. This is more
of an 'rsync-like' mode of operation that can be used to have a directory of
resized images that is kept up-to-date with the source folder.

