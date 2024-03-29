import glob
import os
import platform
from argparse import ArgumentParser

import enlighten
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import numpy as np
from Katna.image import Image as KImage
import cv2
import sys
import logging
from queue import Queue
from threading import Thread

from PIL.Image import Resampling

log = logging.getLogger(__name__)
kimage = None


def pillow_l_to_opencv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)


def pillow_rgb_to_opencv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def opencv_to_pillow(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def process(src, dest, target_res=(1280, 800), max_crop_aspect_delta=0.2, inpaint=False):
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

    if target_aspect * (1 - max_crop_aspect_delta) <= source_aspect <= target_aspect * (1 + max_crop_aspect_delta):
        log.debug(
            f"Source aspect ratio of {source_aspect:.2f} is within {max_crop_aspect_delta:.1%} "
            f"of target aspect ratio {target_aspect:.2f}; using Katna to find best crop"
        )
        # Aspect ratio difference is OK; try smart crop
        if kimage is None:
            kimage = KImage()
        filled = img.resize((filled_w, filled_h), Resampling.LANCZOS)
        crop_list = kimage.crop_image_from_cvimage(
            input_image=pillow_rgb_to_opencv(filled), crop_width=target_w, crop_height=target_h, num_of_crops=1, down_sample_factor=4
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
        # If cropping was not an option, fit the image to the frame and fill in the rest (w/ solid color or inpainting)
        log.debug("Fitting image to frame")

        if inpaint:
            # For inpainting, the background should be a very blurry stretched version of the source image
            stretched = img.resize((target_w, target_h), Resampling.LANCZOS)
            target.paste(stretched, (0, 0))
            target = target.filter(ImageFilter.GaussianBlur(round(max(target_w, target_h) * 0.05)))

        fitted = img.resize((fitted_w, fitted_h), Resampling.LANCZOS)
        x_pos, y_pos = round((target_w - fitted_w) / 2), round((target_h - fitted_h) / 2)
        target.paste(fitted, (x_pos, y_pos))

        if inpaint:
            # Create a stretched blurred version to give the inpainting algorithm some edges to work towards,
            # otherwise it will always fade to solid grey towards the edges. This is especially noticeable
            # in photos with a solid background that is far from black.
            # We don't fully stretch the image to the target aspect ratio, but we also don't leave it
            # as-is; we keep 20% of the original aspect ratio. This finds a bit of a middle ground
            # between keeping sky and ground colors (pure stretching would do this perfectly) and
            # creating large straight (horizontal/vertical) lines in the result (which pure zooming
            # would result in)
            zoom_w, zoom_h = round(filled_w * .2 + target_w * .8), round(filled_h * .2 + target_w * .8)
            zoom_crop_x, zoom_crop_y = round((zoom_w - target_w)/2), round((zoom_h - target_h)/2)
            target = img\
                .resize((zoom_w, zoom_h), Resampling.LANCZOS)\
                .filter(ImageFilter.GaussianBlur(round(max(target_h, target_w) * 0.02)))\
                .crop((zoom_crop_x, zoom_crop_y, zoom_crop_x + target_w, zoom_crop_y + target_h))
            target.paste(fitted, (x_pos, y_pos))

            # Mark everything but the original image and a 1 pixel wide edge all around (containing the 
            # stretched blurred version we created above) for inpainting.
            mask = np.full((target_w, target_h), fill_value=1, dtype=np.uint8)
            mask[x_pos:x_pos+fitted_w, y_pos:y_pos + fitted_h] = 0
            mask[0, :] = 0
            mask[target_w-1, :] = 0
            mask[:, 0] = 0
            mask[:, target_h-1] = 0

            # Do the inpainting
            target = opencv_to_pillow(cv2.inpaint(pillow_rgb_to_opencv(target), mask.T, 3, cv2.INPAINT_TELEA))

            # Blur the inpainting result; we're not getting nor looking for a pixel-perfect continuation of the
            # image, just some soft filler for the edge that doesn't contrast too much with the original
            # scaled-down image in the center.
            target = target.filter(ImageFilter.GaussianBlur(round(max(target_w, target_h) * 0.05)))

            # Paste the original (scaled) image in the center.
            target.paste(fitted, (x_pos, y_pos))

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

    log.info("Collecting images to process...")
    all_files = (
        os.path.join(subdir, image_file)
        for path in paths
        for subdir, _, files in (
            os.walk(path, followlinks=True) if os.path.isdir(path) else [(os.path.dirname(path), 0, [os.path.basename(path)])]
        )
        for image_file in files
    )
    
    dest = os.path.abspath(destination)
    
    jobs = [
        (image_path, os.path.join(dest, os.path.relpath(image_path, base_path)))
        for image_path in all_files if is_image_file(image_path)
    ]

    log.info("Found %s images", len(jobs))
    
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
        "--inpaint", "-i", action='store_true',
        help="Use image inpainting instead of black bars to fill empty spaces when the image is not frame-filling"
    )
    argparse.add_argument(
        "--max-crop-aspect-delta", "-a", type=float, nargs="?", default=.2, 
        help="Upper limit for aspect ratio mismatch between source image and target resolution to apply cropping in stead of fitting and filling. " + 
        "E.g. if a 4:3 source image is processed for a 5:4 target resolution, (4/3)/(5/4) = 1.07. " + 
        "This is within the default mismatch limit of 0.2 (a range of 0.8 - 1.2), so the image will be cropped. " + 
        "A portrait image of 3:4 for the same target resolution will yield (3/4)/(5/4) = 0.6 which exceeds the mismatch limit of 0.2 (0.8 - 1.2), " + 
        "so the image will be fitted and filled in stead."
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
    argparse.add_argument(
        "--overwrite", "-o", action='store_true',
        help="Overwrite existing images. Default is to skip if target exists."
    )
    args = argparse.parse_args()

    try:

        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(format='%(message)s')
        log.setLevel(log_level)

        destination_is_file = False

        paths = args.path
        if platform.system() == 'Windows':
            paths = [p for path in paths for p in glob.glob(path)]

        if not os.path.isdir(args.destination):
            if len(paths) > 1 or args.recurse:
                raise UserInputException("Destination must be an existing directory if multiple input files are specified")
            else:
                destination_is_file = True

        if args.recurse:
            jobs = get_recursive_jobs(paths, args.destination, args.base)
        else:
            jobs = [
                (path, args.destination if destination_is_file else os.path.join(args.destination, os.path.basename(path)))
                for path in paths if is_image_file(path)
            ]

        exceptions = []

        queue = Queue()
        pbar = enlighten.Counter(total=len(jobs), desc='Processing', unit='images')

        def loop():
            while True:
                n, src, dst = queue.get()
                log.info(f"({n+1}/{len(jobs)}) {src} -> {dst}")
                if not args.overwrite and os.path.exists(dst):
                    log.info(f"{dst} already exists, skipping")
                else:
                    if args.recurse and not args.dry_run:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if args.dry_run:
                        log.info("Skipping image processing (--dry-run)")
                    else:
                        try:
                            process(src, dst, args.size, inpaint=args.inpaint, max_crop_aspect_delta=args.max_crop_aspect_delta)
                        except Exception as e:
                            exceptions.append((src, dst, e))
                            if log.isEnabledFor(logging.DEBUG):
                                log.warning(f"Failed to process {src}: {str(e)}", exc_info=sys.exc_info())
                            else:
                                log.warning(f"Failed to process {src}: {str(e)}")
                pbar.update()
                queue.task_done()

        for _ in range(os.cpu_count()):
            Thread(target=loop, daemon=True).start()

        for n, (src, dst) in enumerate(jobs):
            queue.put((n, src, dst))

        queue.join()

        if exceptions:
            log.warning(f"{len(exceptions)} image(s) failed to process:")
            for src, dst, e in exceptions:
                log.warning(f"- {src}: {str(e)}")
            sys.exit(-2)
    
    except UserInputException as e:
        print(str(e))
        print()
        argparse.print_help()
        sys.exit(-1)
    except KeyboardInterrupt:
        log.warning("Aborted by user")
        sys.exit(-3)


if __name__ == '__main__':
    main()

