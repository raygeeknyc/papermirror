import logging
logging.getLogger("").setLevel(logging.INFO)

import time
import io
import sys
import numpy
import cv2
import threading
import Queue

from picamera import PiCamera

from PIL import Image

from inky import InkyWHAT, InkyPHAT
# Set up the correct display and scaling factors
inky_display = InkyPHAT("red")

def overlay_image(display, image, color):
  pixel_count = 0
  for x in range(0, display.WIDTH):
    for y in range(0, display.HEIGHT):
      if image[x,y]:
        display.set_pixel(x,y, color)
        pixel_count += 1
  logging.debug("{} pixels set".format(pixel_count))

RESOLUTION = (600, 400)
CAMERA_ERROR_DELAY_SECS = 1


camera = PiCamera()
camera.resolution = RESOLUTION
camera.vflip = False
image_buffer = io.BytesIO()

def displayImage(display, queue):
    global STOP

    image = None
    previous_image = None
    skipped_images = 0
    last_start = time.time()
    last_report_at = time.time()
    fps = 0
    frame_count = 0
    while not STOP:
        try:
            image = image_queue.get(False)
            skipped_images += 1
            logging.debug("Image queue had an entry")
        except Queue.Empty:
            if not image:
                logging.debug("Empty image queue, waiting")
                skipped_images = 0
            else:
                skipped_images -= 1
                logging.debug("got the most recent image, skipped over {} images".format(skipped_images))
                logging.debug("displaying image %s" % id(image))
		if previous_image:
                    overlay_image(inky_display, previous_image.load(), inky_display.BLACK)
                overlay_image(inky_display, image.load(), inky_display.RED)

                previous_image = image
                image = None
                inky_display.show()
                frame_frequency = time.time() - last_start
                last_start = time.time()
                frame_rate = 1/frame_frequency
                fps += frame_rate
                frame_count += 1
                if last_start - last_report_at >= 1.0:
                    fps /= frame_count
                    frame_count = 0
                    logging.debug("display rate: {} fps".format(fps))
                    fps = 0
                    last_report_at = last_start 

global STOP
STOP = False
try:
    image_queue = Queue.Queue()
    displayer = threading.Thread(target=displayImage, args=(inky_display, image_queue,))
    displayer.start()

    frame_count = 0
    fps = 0
    last_report_at = time.time()
    last_start = time.time()
    s = time.time()
    for _ in camera.capture_continuous(image_buffer, format='jpeg', use_video_port=True):
        try:
            image_buffer.truncate()
            image_buffer.seek(0)
            logging.debug("Camera capture took {}".format(time.time()-s))
        except Exception, e:
            logging.exception("Error capturing image")
            time.sleep(CAMERA_ERROR_DELAY_SECS)
            continue
        s = time.time()
        display_image = Image.open(image_buffer)
        cv2_image = numpy.array(display_image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)
        logging.debug("Image conversion took {}".format(time.time()-s))
        s = time.time()
        cv2_image = cv2.equalizeHist(cv2_image)
        #(thresh, cv2_image) = cv2.threshold(cv2_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # too high contrast
        cv2_image = cv2.resize(cv2_image, (inky_display.width, inky_display.height))
        display_image = Image.fromarray(cv2_image).convert('1')
        image_buffer.seek(0)
        logging.debug("Image processing took {}".format(time.time()-s))
        s = time.time()
        image_queue.put(display_image)
        logging.debug("Image queuing took {}".format(time.time()-s))
        frame_frequency = time.time() - last_start
        last_start = time.time()
        frame_rate = 1/frame_frequency
        fps += frame_rate
        frame_count += 1
        if last_start - last_report_at >= 1.0:
            fps /= frame_count
            frame_count = 0
            logging.debug("ingestion rate: {} fps".format(fps))
            fps = 0
            last_report_at = last_start 
        s = time.time()
except KeyboardInterrupt:
        logging.info("interrupted, exiting")
        STOP = True
        camera.close()
        sys.exit()
