import cv2
import numpy as np
import skimage.measure
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

# higher reduce = less details and more speed, 15-20 is a recommended value
reduce_pixels = 15
color_density_str = "@80GCLft1i;:,. "
max_val = 255
font = PIL.ImageFont.load_default(size=reduce_pixels)
contrast_factor = 128
contrast = 259 * (contrast_factor + 255) / (255 * (259 - contrast_factor))
contrast_image = np.vectorize(lambda x: min(255, max(0, contrast * (x - 128) + 128)), otypes=[np.uint8])
color_density = np.vectorize(lambda x: color_density_str[int(x / max_val * (len(color_density_str) - 1))])


def frame_to_ascii(frame):
    return color_density(contrast_image(frame))


def ascii_to_image(ascii_frame):
    w, h = ascii_frame.shape
    image = PIL.Image.new('RGB', (h * reduce_pixels, w * reduce_pixels), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(image)
    for i in range(w):
        for j in range(h):
            draw.text((j * reduce_pixels, i * reduce_pixels,), ascii_frame[i, j], font=font, fill=(255, 255, 255))

    return np.array(image)


def main():
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = skimage.measure.block_reduce(frame, (reduce_pixels, reduce_pixels), np.max)
        ascii_map = frame_to_ascii(frame)
        image = ascii_to_image(ascii_map)

        if not ret:
            break
        cv2.imshow("ASCII camera (q - quit)", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    camera.release()


if __name__ == "__main__":
    main()
