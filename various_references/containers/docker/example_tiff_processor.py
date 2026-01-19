#!/usr/bin/env python3
"""
Simple TIFF image processor using tifffile
"""
import tifffile
import numpy as np

def main(image_path: str = '/temp/input.tif'):
    print(f"Reading TIFF image from: {image_path}")

    # Read the TIFF file
    image = tifffile.imread(image_path)

    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min value: {np.min(image)}")
    print(f"Image max value: {np.max(image)}")
    print(f"Image mean value: {np.mean(image):.2f}")

    # Do some simple processing
    if len(image.shape) == 2:
        # Grayscale image
        print("\nThis is a grayscale image")
        height, width = image.shape
        print(f"Dimensions: {width} x {height} pixels")
    elif len(image.shape) == 3:
        # Color or multi-channel image
        if image.shape[2] <= 4:
            # Likely RGB or RGBA
            print(f"\nThis is a color image with {image.shape[2]} channels")
            height, width, channels = image.shape
            print(f"Dimensions: {width} x {height} pixels, {channels} channels")
        else:
            # Multi-page or z-stack
            print(f"\nThis is a multi-page/z-stack image with {image.shape[0]} slices")

    # Write some output
    output_path = "/temp/output.txt"
    with open(output_path, 'w') as f:
        f.write(f"Image Analysis Results\n")
        f.write(f"=====================\n")
        f.write(f"Input file: {image_path}\n")
        f.write(f"Shape: {image.shape}\n")
        f.write(f"Data type: {image.dtype}\n")
        f.write(f"Min: {np.min(image)}\n")
        f.write(f"Max: {np.max(image)}\n")
        f.write(f"Mean: {np.mean(image):.2f}\n")

    print(f"\nResults written to: {output_path}")
    print("Processing complete!")


if __name__ == "__main__":
    import sys
    print(f"args: {sys.argv}")
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
