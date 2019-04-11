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
      if not new_pixels[x,y]:
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
FRAME_DISPLAY_DELAY_SECS = 1
FRAME_DISPLAY_SHIFT_SECS = 1
PIXEL_SHIFT_SENSITIVITY = 20  # The threshold within which a pixel's channel changes are ignored


camera = PiCamera()
camera.resolution = RESOLUTION
camera.vflip = False
image_buffer = io.BytesIO()
MOTION_DETECT_SAMPLE_PCT = 0.10
MOTION_DETECTION_DELTA_THRESHOLD_PCT = 0.25
COMPARE_CHANNEL = 1  # compare the green channel of RGB images

def motionDetected(cv2_image1, cv2_image2, pixel_tolerance_percent, sample_percentage=MOTION_DETECT_SAMPLE_PCT):
	if cv2_image1 is None and cv2_image2 is None:
		logging.debug("no images")
		return False
	if cv2_image1 is None or cv2_image2 is None:
		logging.debug("only one image")
		return True
	if cv2_image1 is cv2_image2:
		logging.error("same image")
		return False
	sample_delta_threshold = pixel_tolerance_percent * sample_percentage
        s=time.time()
	height, width, _ = cv2_image1.shape
	logging.debug("h,w: {},{}".format(height,width))
        current_pixels = cv2_image2
        prev_pixels = cv2_image1
        pixel_step = int((width * height)/(sample_percentage * width * height))
	pixel_tolerance = int(sample_delta_threshold * width * height)
	logging.debug("delta threshold: {} pixel tolerance: {}".format(sample_delta_threshold, pixel_tolerance))
	sampled_pixels = 0
        changed_pixels = 0
        for pixel_index in xrange(0, width*height, pixel_step):
	    sampled_pixels += 1
            if abs(int(current_pixels[pixel_index%height,pixel_index/height][COMPARE_CHANNEL]) - int(prev_pixels[pixel_index%height,pixel_index/height][COMPARE_CHANNEL])) >= PIXEL_SHIFT_SENSITIVITY:
                changed_pixels += 1
                if changed_pixels > pixel_tolerance:
                  logging.debug("Image diff {} of {}".format(changed_pixels, sampled_pixels))
                  return True
        logging.debug("Images equal: {} diff of {} sampled".format(changed_pixels, sampled_pixels)) 
        return False

def displayTransition(inky_display, previous_image, image):
	if previous_image != None:
		overlay_image(inky_display, previous_image, inky_display.RED)
		inky_display.show()
		time.sleep(FRAME_DISPLAY_SHIFT_SECS)
	if image != None:
		overlay_image(inky_display, image, inky_display.BLACK)
		inky_display.show()
		time.sleep(FRAME_DISPLAY_SHIFT_SECS)
		overlay_image(inky_display, image, inky_display.BLACK, inky_display.WHITE)
		inky_display.show()
		time.sleep(FRAME_DISPLAY_DELAY_SECS)

def showImages(display, queue):
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
                logging.debug("image")
		displayTransition(inky_display, previous_image, image)
               	previous_image = image
               	image = None
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
    displayer = threading.Thread(target=showImages, args=(inky_display, image_queue,))
    displayer.start()

    frame_count = 0
    fps = 0
    last_report_at = time.time()
    last_start = time.time()
    s = time.time()
    prev_cv2_rgb_image = None
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
        cv2_rgb_image = numpy.array(frame)
        logging.debug("Image conversion took {}".format(time.time()-s))
        s = time.time()
	if motionDetected(prev_cv2_rgb_image, cv2_rgb_image, MOTION_DETECTION_DELTA_THRESHOLD_PCT):
		logging.debug("motion detected")
		prev_cv2_rgb_image = cv2_rgb_image
        	cv2_grayscale_image = cv2.cvtColor(cv2_rgb_image, cv2.COLOR_RGB2GRAY)
        	cv2_grayscale_image = cv2.equalizeHist(cv2_grayscale_image)
        	image_center = tuple(numpy.array(cv2_grayscale_image.shape[1::-1]) / 2)
        	rot_mat = cv2.getRotationMatrix2D(image_center, 270, 1.0)
        	cv2_grayscale_image = cv2.warpAffine(cv2_grayscale_image, rot_mat, cv2_grayscale_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        	new_width =  cv2_grayscale_image.shape[0]
        	new_height =  int(new_width * (1.0 * new_width / cv2_grayscale_image.shape[1]))
        	logging.debug("n_w,n_h: {},{}".format(new_width,new_height))
        	x_margin = (cv2_grayscale_image.shape[1] - new_width)/2
        	y_margin = (cv2_grayscale_image.shape[0] - new_height)/2
        	logging.debug("x_m,y_m: {},{}".format(x_margin,y_margin))
        	cropped_grayscale_image = cv2_grayscale_image[y_margin:y_margin + new_height, x_margin:x_margin + new_width]
        	cropped_grayscale_image = cv2.resize(cropped_grayscale_image, (inky_display.WIDTH, inky_display.HEIGHT))
        	b_w_image = Image.fromarray(cropped_grayscale_image).convert('1')
        	logging.debug("Image processing took {}".format(time.time()-s))
        	s = time.time()
        	image_queue.put(b_w_image)
        	logging.debug("Image queuing took {}".format(time.time()-s))
        image_buffer.seek(0)
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
        logging.warning("interrupted, exiting")
        STOP = True
        camera.close()
        sys.exit()
