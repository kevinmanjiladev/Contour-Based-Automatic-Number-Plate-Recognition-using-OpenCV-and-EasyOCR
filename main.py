import cv2
import numpy as np
import imutils
import easyocr

# Image read
img=cv2.imread("Car_number_plate2.jpg")

# Resize image
img=cv2.resize(img,(600,400))

# Covert to gray scale
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# Apply filters and find edges for localization

# bilateral filter for NOISE REDUCTION
bfilter=img.copy()
for _ in range(7):
    bfilter=cv2.bilateralFilter(img,11,17,17)

# Edge detection using canny
edged=cv2.Canny(bfilter,100,200)




# Find contours and Apply Mask
keypoints=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(keypoints)
contours=sorted(contours,key=cv2.contourArea,reverse=True)[:20]

location=None
for contour in contours:
    approx=cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx)==4:
        location=approx
        break


mask=np.zeros(gray_img.shape,np.uint8)
if location is None:
    print("⚠ No number plate detected")
    exit()
new_image=cv2.drawContours(mask,[location],0,255,-1)
new_image=cv2.bitwise_and(img,img,mask=mask)

(x,y)=np.where(mask==255)
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))
cropped_img=gray_img[x1:x2+1,y1:y2+1]

reader=easyocr.Reader(['en'])
result=reader.readtext(cropped_img)
if len(result)>0:
    text=f"Car number: {result[0][1]}"
    cv2.putText(img,text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),8)
    cv2.putText(img,text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.rectangle(img,(y1,x1),(y2,x2),(0,255,0),3)
else:
    text="No Plate Detected!"
    cv2.putText(img,text,(270,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),8)
    cv2.putText(img,text,(270,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


print("Number Plate:", text)
print(location)
cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
