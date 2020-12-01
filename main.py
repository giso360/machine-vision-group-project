import cv2


videocapture = cv2.VideoCapture('./data/YouTube_test_set_video.mp4')
success, image = videocapture.read()
success = True
count = 0
# capture frame every 500ms
while success:
  videocapture.set(cv2.CAP_PROP_POS_MSEC, count*500)
  success, image = videocapture.read()
  cv2.imwrite("./data/frames/frame%d.jpg" % count, image)
  print(success)
  count = count + 1
  if count == 201:
    break



print("OK")