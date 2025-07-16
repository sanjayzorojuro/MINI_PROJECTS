from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#load the info and convert it into grayscale and put in a numpy array
image=Image.open(r"C:\Python\PHOTO_MANI\zoro.jpg").convert("L")
img_arr=np.array(image)

#show original image
plt.figure(figsize=(15,4))
plt.subplot(1,4,1)
plt.imshow(img_arr,cmap="gray")
plt.title("ORIGINAL")
plt.axis('off')

#invert the image
inverted=255-img_arr
plt.subplot(1,4,2)
plt.imshow(inverted,cmap="gray")
plt.title("INVERTED")
plt.axis('off')

#brightness adjustment
def addjust_brightness(img,value):
    return np.clip(img+value,0,255)

bright=addjust_brightness(img_arr,50)
plt.subplot(1,4,3)
plt.imshow(bright,cmap='gray')
plt.title("BRIGHTNESS")
plt.axis('off')

#thresholding
thresh_val=128
thresh=(img_arr > thresh_val) * 255
thresh=thresh.astype(np.uint8)
plt.subplot(1,4,4)
plt.imshow(thresh,cmap="gray")
plt.title(f"thresh @ {thresh_val}")
plt.axis('off')

plt.tight_layout() 
plt.show()

#save the processed image back from np array to jpg format
Image.fromarray(inverted).save("inverted.jpg")
Image.fromarray(bright).save("brightness.jpg")
Image.fromarray(thresh).save("threshholding.jpg")

