import cv2
import numpy as np
import skimage.measure
import pygame

reduce_pixels = 8
color_density_str = "@80GCLft1i;:,. "
max_val = 255
contrast_factor = 128
contrast = 259 * (contrast_factor + 255) / (255 * (259 - contrast_factor))
contrast_image = np.vectorize(lambda x: min(255, max(0, contrast * (x - 128) + 128)), otypes=[np.uint8])
color_density = np.vectorize(lambda x: color_density_str[int(x / max_val * (len(color_density_str) - 1))])

pygame.init()
font = pygame.font.SysFont("monospace", reduce_pixels,  bold=True)
screen = pygame.display.set_mode((1366, 768), pygame.DOUBLEBUF)
pygame.display.set_caption("ASCII Camera")


def frame_to_ascii(frame):
    return color_density(contrast_image(frame))


def ascii_to_image(ascii_frame):
    w, h = ascii_frame.shape
    screen.fill((0, 0, 0))
    for i in range(w):
        for j in range(h):
            char = ascii_frame[i, j]
            text_surface = font.render(char, True, (255, 255, 255))
            screen.blit(text_surface, (j * reduce_pixels, i * reduce_pixels))
    pygame.display.flip()


def main():
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = skimage.measure.block_reduce(frame, (reduce_pixels, reduce_pixels), np.max)
        ascii_map = frame_to_ascii(frame)
        ascii_to_image(ascii_map)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.release()
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                camera.release()
                pygame.quit()
                return

if __name__ == "__main__":
    main()