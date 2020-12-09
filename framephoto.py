import os
from argparse import ArgumentParser
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import numpy as np
from Katna.image import Image as KImage
import cv2
import sys
import logging


log = logging.getLogger(__name__)
kimage = None


def process(src, dest, target_res=(1280, 800), max_crop_aspect_diff=0.2):
    global kimage

    img = Image.open(src)
    img = ImageOps.exif_transpose(img)

    target_w, target_h = target_res
    source_w, source_h = img.width, img.height

    target_aspect = target_w/target_h
    source_aspect = source_w/source_h

    blur_radius = max(target_w, target_h) * .05 

    target = Image.new('RGB', (target_w, target_h), (0, 0, 0))

    if target_aspect > source_aspect:
        # Source is taller than target
        fitted_scale = target_h / source_h
        fitted_w = round(source_w * fitted_scale)
        fitted_h = target_h
        
        filled_scale = target_w / source_w
        filled_w = target_w
        filled_h = round(source_h * filled_scale)
    else:
        # Source is wider than target
        fitted_scale = target_w / source_w
        fitted_w = target_w
        fitted_h = round(source_h * fitted_scale)

        filled_scale = target_h / source_h
        filled_w = round(source_w * filled_scale)
        filled_h = target_h

    image_pasted = False

    if target_aspect * (1 - max_crop_aspect_diff) <= source_aspect <= target_aspect * (1 + max_crop_aspect_diff):
        log.debug(
            f"Source aspect ratio of {source_aspect:.2f} is within {max_crop_aspect_diff:.1%} "
            f"of target aspect ratio {target_aspect:.2f}; using Katna to find best crop"
        )
        # Aspect ratio difference is OK; try smart crop
        if kimage is None:
            kimage = KImage()
        filled = img.resize((filled_w, filled_h), Image.ANTIALIAS)
        filled_opencv = cv2.cvtColor(np.array(filled), cv2.COLOR_RGB2BGR)
        crop_list = kimage.crop_image_from_cvimage(
            input_image=filled_opencv, crop_width=target_w, crop_height=target_h, num_of_crops=1, down_sample_factor=4
        )

        if len(crop_list) > 0:
            crop = crop_list[0]
            x, y = crop.x, crop.y
            log.debug(f"Katna found crop of source {filled_w}x{filled_h} to {x},{y}+{target_w},{target_h}")
            if x + target_w > filled_w:
                x = filled_w - target_w
            if x < 0: x = 0
            if y + target_h > filled_h:
                y = filled_h - target_h
            if y < 0:
                y = 0
            target.paste(filled, (-x, -y))
            image_pasted = True
        else:
            log.warning("Katna found no appropriate crop")

    if not image_pasted:
        log.debug("Fitting image to frame")
        # As final solution, just paste in the middle of the target
        fitted = img.resize((fitted_w, fitted_h), Image.ANTIALIAS)
        target.paste(fitted, (round((target_w - fitted_w) / 2), round((target_h - fitted_h) / 2)))

    target.save(dest, "JPEG", quality=85)


class UserInputException(Exception):
    pass


def res_str(string):
    w, h = [int(v.strip()) for v in string.split(',')]
    if w < 1 or h < 1:
        raise ValueError("Resolution must be positive")
    return w, h


def is_image_file(file):
    if os.path.isfile(file) and file[-4:].lower() in ('.jpg', '.png'):
        return True


def get_recursive_jobs(source_paths, destination, base_path):
    paths = [os.path.abspath(path) for path in source_paths]
    
    if base_path is None:
        base_path = os.path.commonpath(paths)
        log.info(f"Base path detected as {base_path}")
    
    all_files = (
        os.path.join(subdir, image_file)
        for path in paths
        for subdir, _, files in (
            os.walk(path) if os.path.isdir(path) else [(os.path.dirname(path), 0, [os.path.basename(path)])]
        )
        for image_file in files
    )
    
    dest = os.path.abspath(destination)
    
    jobs = [
        (image_path, os.path.join(dest, os.path.relpath(image_path, base_path)))
        for image_path in all_files if is_image_file(image_path)
    ]
    
    return jobs


def main():
    argparse = ArgumentParser(
        description="Make a nice screen-filling image of a fixed resolution from a photo of any size or aspect ratio"
    )
    argparse.add_argument(
        "path", nargs="+", type=str,
        help="The file(s) or folder(s) (in case of --recurse) to transform"
    )
    argparse.add_argument(
        "--recurse", "-r", action='store_true',
        help="Treat the input path(s) as folder and process all viable images within it. "
             "Skip images for which the target file already exists. "
             "Replicate the source directory structure in the target folder."
    )
    argparse.add_argument(
        "--base", "-b", type=str, default=None,
        help="Use with --recurse (-r); defines the base directory that will correspond to the destination directory. "
             "Must be a parent of ALL specified input paths. The default is to auto-detect a common parent for all specified input paths."
    )
    argparse.add_argument(
        "--size", "-s", type=res_str, default=(1280, 800),
        help="Resolution to fit the image to. Default 1280,800"
    )
    argparse.add_argument(
        "destination", type=str,
        help="The destination folder into which to place the resulting image files"
    )
    argparse.add_argument(
        "--dry-run", "-d", action='store_true',
        help="Don't process images, just output what would be done."
    )
    argparse.add_argument(
        "--verbose", "-v", action='store_true',
        help="Enable verbose mode (debug logging)"
    )
    args = argparse.parse_args()

    try:

        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(format='%(message)s')
        log.setLevel(log_level)

        destination_is_file = False

        if not os.path.isdir(args.destination):
            if len(args.path) > 1 or args.recurse:
                raise UserInputException("Destination must be an existing directory if multiple input files are specified")
            else:
                destination_is_file = True

        if args.recurse:
            jobs = get_recursive_jobs(args.path, args.destination, args.base)
        else:
            jobs = [
                (path, args.destination if destination_is_file else os.path.join(args.destination, os.path.basename(path)))
                for path in args.path if is_image_file(path)
            ]

        for n, (src, dst) in enumerate(jobs):
            log.info(f"({n+1}/{len(jobs)}) {src} -> {dst}")
            if args.recurse and not args.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            if args.dry_run:
                log.info("Skipping image processing (--dry-run)")
            else:
                process(src, dst, args.size)
    
    except UserInputException as e:
        print(str(e))
        print()
        argparse.print_help()
        sys.exit(-1)


if __name__ == '__main__':
    main()
