import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread("C:/Users/user/Desktop/CODE/CODE/Yeni qovluq/developia/PROJECTS/license plate detection/plate.jpg")

image = imutils.resize(image, width=500)

cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray scaled", gray)
cv2.waitKey(0)

gray = cv2.bilateralFilter(gray, 11, 17,17)
cv2.imshow("smoother gray scaled", gray)
cv2.waitKey(0)

edged = cv2.Canny(gray, 170,200)
cv2.imshow("edged", edged)
cv2.waitKey(0)

cnts , new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image1 = image.copy()
cv2.drawContours(image1 , cnts , -1, (0,255,0),3)
cv2.imshow("edged after contouring", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0,255,0),3)
cv2.imshow("top 30 contours", image2)
cv2.waitKey(0)

count = 0
name = 1
for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
    if(len(approx) == 4):
        NumberPlateCount = approx
        x , y , w , h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        cv2.imwrite(str(name)+'.jpg',crp_img)
        name += 1
        break
cv2.drawContours(image, [NumberPlateCount], -1, (0,255,0),3)
cv2.imshow("final image", image)
cv2.waitKey(0)

crop_img_loc = "croppedimg.jpg"
cv2.imshow("Cropped img", cv2.imread(crop_img_loc))
cv2.waitKey(0)

text = pytesseract.image_to_string(crop_img_loc,lang='eng')
print("plate: ", text)
cv2.waitKey(0)