import argparse
import os
from glob import glob

import cv2
from natsort import natsorted
from tqdm import tqdm

DEFAULT_IMAGE_SIZE = (1024, 768)


def tuple_type(arg_string):
    try:
        # Parse the input string as a tuple
        parsed_tuple = tuple(map(int, arg_string.strip("()").split(",")))
        return parsed_tuple
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {arg_string}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input image or dir path"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "-s",
        "--image_size",
        type=tuple_type,
        default=f"({DEFAULT_IMAGE_SIZE[0]}, {DEFAULT_IMAGE_SIZE[1]})",
        help="Output image size as a (width, height) tuple",
    )

    args = parser.parse_args()

    if os.path.isfile(args.input):
        input_paths = [args.input]
    else:
        input_paths = natsorted(
            glob(os.path.join(args.input, "*.jpg"))
            + glob(os.path.join(args.input, "*.png"))
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_path in tqdm(input_paths):
        img_filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, args.image_size, interpolation=cv2.INTER_LINEAR)
        output_filename = os.path.join(args.output_dir, img_filename)
        cv2.imwrite(output_filename, img)


if __name__ == "__main__":
    main()
