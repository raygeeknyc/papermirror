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

from PIL import Image, ImageDraw

from inky import InkyWHAT, InkyPHAT
# Set up the correct display and scaling factors
inky_display = InkyPHAT("red")

def overlay_image(display, image, color, background_color = None):
  logging.debug("Color: {} bg color: {}".format(color, background_color))
  pixel_count = 0
  bg_pixel_count = 0
  width, height = image.size
  if image.size != (display.WIDTH, display.HEIGHT):
    raise ValueError("Image size {} does not match the display {}".format(image.size, (display.WIDTH, display.HEIGHT)))
  new_pixels = image.load()
  for x in range(0, width):
    for y in range(0, height):
      if new_pixels[x,y]:
        display.set_pixel(x, y, color)
        pixel_count += 1
      else:
        if background_color != None:
          display.set_pixel(x, y, background_color)
          bg_pixel_count += 1
  logging.debug("{} pixels set".format(pixel_count))
  logging.debug("{} bg pixels set".format(bg_pixel_count))

RESOLUTION = (320, 240)
CAMERA_ERROR_DELAY_SECS = 1
FRAME_DISPLAY_DELAY_SECS = 5


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
                    logging.debug("previous_image")
                    overlay_image(inky_display, previous_image, inky_display.RED, inky_display.WHITE)
                logging.debug("image")
                overlay_image(inky_display, image, inky_display.BLACK)

                inky_display.show()
                previous_image = image
                image = None
		time.sleep(FRAME_DISPLAY_DELAY_SECS)
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
        frame = Image.open(image_buffer)
        cv2_image = numpy.array(frame)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)
        logging.debug("Image conversion took {}".format(time.time()-s))
        s = time.time()
        cv2_image = cv2.equalizeHist(cv2_image)
        image_center = tuple(numpy.array(cv2_image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 270, 1.0)
        cv2_image = cv2.warpAffine(cv2_image, rot_mat, cv2_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        new_width =  cv2_image.shape[0]
        new_height =  int(new_width * (1.0 * new_width / cv2_image.shape[1]))
        logging.debug("n_w,n_h: {},{}".format(new_width,new_height))
        x_margin = (cv2_image.shape[1] - new_width)/2
        y_margin = (cv2_image.shape[0] - new_height)/2
        logging.debug("x_m,y_m: {},{}".format(x_margin,y_margin))
        crop_image = cv2_image[y_margin:y_margin + new_height, x_margin:x_margin + new_width]
        crop_image = cv2.resize(crop_image, (inky_display.WIDTH, inky_display.HEIGHT))
        frame = Image.fromarray(crop_image).convert('1')
        image_buffer.seek(0)
        logging.debug("Image processing took {}".format(time.time()-s))
        s = time.time()
        image_queue.put(frame)
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
