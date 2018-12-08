#!/usr/bin/env python

import argparse
from PIL import Image, ImageDraw
from inky import InkyWHAT, InkyPHAT
import cv2
import numpy

# Set up the correct display and scaling factors

inky_display = InkyPHAT("red")

def overlay_image(display, image, color):
  pixel_count = 0
  for x in range(0, display.WIDTH):
    for y in range(0, display.HEIGHT):
      if image[x,y]:
        display.set_pixel(x,y, color)
        pixel_count += 1
  print("{} pixels set".format(pixel_count))

pil_image = Image.open('images/test_1.jpg')
cv2_image = numpy.array(pil_image)
cv2_image = cv2.resize(cv2_image, (inky_display.WIDTH, inky_display.HEIGHT))
cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)
(thresh, cv2_image) = cv2.threshold(cv2_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
flattened_image = Image.fromarray(cv2_image).convert('1')
overlay_image(inky_display, flattened_image.load(), inky_display.BLACK)

pil_image = Image.open('images/test_2.jpg')
cv2_image = numpy.array(pil_image)
cv2_image = cv2.resize(cv2_image, (inky_display.WIDTH, inky_display.HEIGHT))
cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)
(thresh, cv2_image) = cv2.threshold(cv2_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
flattened_image = Image.fromarray(cv2_image).convert('1')
overlay_image(inky_display, flattened_image.load(), inky_display.RED)
print("size: {},{}".format(inky_display.WIDTH, inky_display.HEIGHT))
inky_display.show()
