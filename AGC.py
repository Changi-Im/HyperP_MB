import cv2
import math
import numpy as np


def AGC(img):
    img = img/255
    mean, std_1 = cv2.meanStdDev(img, mask=None)
    D = abs((mean + 2*std_1)-(mean-2*std_1))
    
    # for low contrast image
    if D <= 1/3:
        gamma = -math.log2(std_1)
    # for high contrast or moderate contrast image
    else:
        gamma = np.exp((1-(mean + std_1))/2)
    
    k = pow(img,gamma) + (1 - pow(img,gamma)) * pow(mean,gamma)
    c = 1 / (1 + Heaviside(0.5-mean) * (k-1))
    out = c * pow(img,gamma)
    out = out*255
    out = out.astype('uint8')
    
    return out
    
    
def Heaviside(x):
    if x <= 0:
        return 0
    else:
        return 1

def gamma_generator(img):
    img = img/255
    mean, std_1 = cv2.meanStdDev(img, mask=None)
    D = abs((mean + 2*std_1)-(mean-2*std_1))
    # for low contrast image
    if D <= 1/3:
        gamma = -math.log2(std_1)
    # for high contrast or moderate contrast image
    else:
        gamma = math.exp((1-(mean + std_1))/2)
    return gamma

def lambda_generator(source1, source2, target):
    gamma1 = gamma_generator(source1)
    gamma2 = gamma_generator(source2)
    gamma = gamma_generator(target)
    
    d1 = abs(gamma1 - gamma)
    d2 = abs(gamma2 - gamma)
    
    lambda1 = 2*d1/(d1+d2)
    lambda2 = 2*d2/(d1+d2)
    
    return lambda1, lambda2
    
def main():
    img = cv2.imread("../DB/19_input1.jpg",0)
    img1 = cv2.imread("../DB/19_input2.jpg",0)
    img2 = cv2.imread("../DB/19_target.jpg",0)
    gamma = gamma_generator(img)
    gamma1 = gamma_generator(img1)
    gamma2 = gamma_generator(img2)
    lambda1, lambda2 = lambda_generator(img,img1,img2)
    print("underexposure - centerexposure gamma : ", abs(gamma2 - gamma))
    print("overexposure - centerexposure gamma : ", abs(gamma2 - gamma1))
    print("lambda1 : ", lambda1)
    print("lambda2 : ", lambda2)
    
    #cv2.imshow("0",out)
    #cv2.imshow("1",out1)
    #cv2.imshow("2",out2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()