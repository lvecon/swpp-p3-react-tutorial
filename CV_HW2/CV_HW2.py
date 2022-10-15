import math
import glob
import numpy as np
from PIL import Image

# parameters

datadir = './data'
resultdir='./results'

# you can calibrate these parameters
sigma=2
threshold=0.03
rhoRes=150
thetaRes=360
nLines=20

def zero_pad(img, k):
    # TODO

    result = np.zeros((len(img)+2*k, len(img[0])+2*k))
    for i in range(len(img)):
        for j in range(len(img[0])):
            result[i+k][j+k] = img[i][j]

#    for i in range(k):
#        for j in range(len(img)):
#            result[k-i-1][j+k] = result[k+i][j+k]
#            result[k-i-1][k+len(img[0])+i]=

#    for i in range(k):
#        for j in range(len(img[0])):
#            result[k-i+1][j+k] = result[k+i+1][j]
#            result[len(img)+k+i+1][j+k] = result[len(img)+k-i][j+k]

    return np.pad(img, k, mode="reflect")


def convolve(img, kernel):
    # TODO
    kernel_length = len(kernel)//2
    in_image = zero_pad(img, kernel_length)
    result = np.zeros((len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            temp = 0
            for k in range(len(kernel)):
                for l in range(len(kernel)):
                    temp = temp+kernel[k][l]*in_image[i-k+2*kernel_length][j-l+2*kernel_length]
            result[i][j]=temp
    return result

def get_gaussian(k, sigma):
    # TODO
    kernel = np.zeros((2*k+1,2*k+1))
    temp = 0
    for i in range(2*k+1):
        for j in range(2*k+1):
            kernel[i][j] = (1/(2*math.pi*sigma**2))*math.exp(-(((k-i)**2+(k-j)**2)/(2*sigma**2)))
            temp = temp+kernel[i][j]
    kernel = kernel/temp
    return kernel


def ConvFilter(Igs, G):
    # TODO ...
    Iconv = convolve(Igs, G)
    return Iconv

def EdgeDetection(Igs, sigma):
    # TODO ...
    G = get_gaussian(3, sigma) #k를 그냥 3으로 잡아봄
    Igs = zero_pad(Igs, 3)
    Iconv = ConvFilter(Igs, G)
    xSobel = np.array([[-1, 0, 1], [-2,0,2],[-1,0,1]])
    ySobel = np.array([[1, 2, 1], [0,0,0], [-1,-2,-1]])
    Ix = ConvFilter(Iconv, xSobel)
    Iy = ConvFilter(Iconv, ySobel)
    Iy[np.where(Iy==0)] = 0.0001

    Im = np.power(np.power(Ix,2)+np.power(Iy,2),(1/2))
    Io = np.arctan(np.abs(Ix)/np.abs(Iy))

    for i in range(1, len(Io)-1):
        for j in range(1, len(Io[0])-1):
            q = Im[i-1][j]
            r = Im[i+1][j]
            s = Im[i][j+1]
            t = Im[i][j-1]

            x = Im[i][j]
            if (x<q or x<r or x<s or x<t):
                Im[i][j]=0
    return Im, Io, Ix, Iy

def HoughTransform(Im,threshold, rhoRes, thetaRes):
    # TODO ...
    rho_max = math.sqrt(math.pow(len(Im), 2)+math.pow(len(Im[0]), 2))
    H = np.zeros((2*rhoRes, thetaRes))
    for i in range (len(Im)):
        for j in range(len(Im[0])):
            if Im[i][j] > threshold :
                for k in range (thetaRes):
                    temp = i*math.cos(k*thetaRes*math.pi)+j*math.sin(k*thetaRes*math.pi)
                    temp = round(temp*rhoRes/rho_max)+rhoRes
                    H[temp][k] = H[temp][k]+1
    a = np.amax(H)
    return H/a

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...

    def findMax(arr):
        max = 0
        x=0
        y=0
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if arr[i][j]>max:
                    max = arr[i][j]
                    x=i
                    y=j
        return x, y
    lRho = list()
    lTheta = list()
    for i in range(nLines):
        a, b = findMax(H)
        lRho.append(a)
        lTheta.append(b)
        for i in range(-3,4):
            for j in range(-3,4):
                if (a+i >=0 and a+i<len(H) and b+j>=0 and b+j<len(H[0])):
                    H[a+i][b+j]=0
    print(lRho, lTheta)

    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im, threshold):
    # TODO ...


    return l

def drawLines(img, lRho, lTheta, rhoRes, thetaRes):
    img = np.array(img)
    img = img / 255.
    line_img = np.zeros((len(img), len(img[0])))
    for i in range(len(lRho)):
        a = lRho[i]*math.pi/thetaRes
        b = lTheta[i]
        slope = -math.cos(a)/math.sin(a)
        intercept = (b-rhoRes*math.sqrt(math.pow(len(img),2)+math.pow(len(img[0]),2)))/(rhoRes*math.sin(a))
        print(slope, intercept)
        for j in range(len(img)):
            y = slope*j+intercept
            if (y>=0 and y<len(img)):
                line_img[j][y] = 1
                print(j, y)
    Image.fromarray(np.uint8(line_img * 255)).save(f'output1-3.jpg')

def main():
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
      #  drawLines(img, lRho, lTheta, rhoRes, thetaRes)


        #l = HoughLineSegments(lRho, lTheta, Im, threshold)

        # saves the outputs to files
  #      Image.fromarray(np.uint8(Im * 255)).save(img_path+f'output1.jpg')
   #     Image.fromarray(np.uint8(H * 255)).save(img_path+f'output1-2.jpg')

        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()
