import cv2
import numpy as np

imageA = cv2.imread('A.png', cv2.IMREAD_UNCHANGED)  
imageB = cv2.imread('B.png', cv2.IMREAD_GRAYSCALE)  

# Test of fusion method (results are mixed)
if imageA is None or imageB is None:
    print("Error: Could not load imgs.")
else:
    imageA = cv2.resize(imageA, (imageB.shape[1], imageB.shape[0]))

    alpha_channel = imageA[:, :, 3].astype(np.float32) / 255.0  

    imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)

    alpha_value = 0.6 

    result = (BGR_channels * alpha_value + imageB * (1 - alpha_value)).astype(np.uint8)

    cv2.imshow('Superimposed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('result.png', result)
