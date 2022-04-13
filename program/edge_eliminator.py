import cv2
import numpy as np

img = cv2.imread("1.jpg")
img_crop = img[74:954,517:1397,:] # height, width, depth

img_h, img_w = img_crop.shape[:2]
print(img_crop.shape)
resize = cv2.resize(img_crop, (640,640), interpolation = cv2.INTER_AREA)

#img_crop = cv2.resize(img_crop, (1920,1080), interpolation = cv2.INTER_AREA)

#define circles
#radius = 440
#xc = 440
#yc = 440

# inverse mask
#inv_mask = np.zeros_like(img_crop)
#inv_mask = cv2.circle(inv_mask, (xc, yc), radius, (255,255,255), -1)
#ret, original_mask = cv2.threshold(inv_mask, 10, 255, cv2.THRESH_BINARY_INV)

# put mask into alpha channel of input -- doesn't work
#result = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
#result[:,:,3] = inv_mask[:,:,0]

# masking process -- doesn't work either
#roi = img[74:955,517:1398]
#roi_bg = cv2.bitwise_and(roi, roi, mask = original_mask)
#roi_fg = cv2.bitwise_and(img_crop, img_crop, mask = inv_mask)
#dst = cv2.add(roi_bg, roi_fg)

#img_crop[74:955,517:1398,:] = dst

#cv2.imshow("image", crop)
#cv2.imshow("Inverse Mask", inv_mask)
#cv2.imshow("Mask", original_mask)
cv2.imshow("result", img_crop)

cv2.waitKey(0)

cv2.destroyAllWindows()
