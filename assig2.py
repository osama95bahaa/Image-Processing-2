import cv2
import numpy as np
import timeit

def histogram(img):

    histogramList = np.zeros(256, dtype=np.int)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            histogramList[img[i,j]] +=1

    cumulativeHistogramList = np.cumsum(histogramList)
    #print(histogramList)
    #print(cumulativeHistogramList)

    scalingFactorHist = 512/np.amax(histogramList)
    scalingFactorCumHist = 512/np.amax(cumulativeHistogramList)

    result = np.zeros((512,1024,1),dtype=np.int)

    for x in range(0, 1020, 4):
        y1 = histogramList[int(x / 4)]
        y1 = 512 - y1 * scalingFactorHist
        result[int(y1): 512, x + 5] = 150
        result[int(y1): 512, x + 1 + 5] = 150
        y2 = cumulativeHistogramList[int(x / 4)]
        y2 = 512 - y2 * scalingFactorCumHist
        result[int(y2): 512, x + 2 + 5] = 200

    return result


#Q1

def cameramanH():
    img = cv2.imread('cameraman.png', cv2.IMREAD_COLOR)
    img = histogram(img)
    cv2.imwrite('resultImages/Q1/cameramanHistogram.png', img)
    cv2.imshow('Cameraman Histogram & cumHisto', img)

def batmanH():
    img = cv2.imread('bat.png', cv2.IMREAD_COLOR)
    img = histogram(img)
    cv2.imwrite('resultImages/Q1/batmanHistogram.png', img)
    cv2.imshow('Batman Histogram & cumHisto', img)

def fogH():
    img = cv2.imread('fog.png', cv2.IMREAD_COLOR)
    img = histogram(img)
    cv2.imwrite('resultImages/Q1/fogHistogram.png', img)
    cv2.imshow('Fog Histogram & cumHisto', img)

def fogAndNoiseH():
    img = cv2.imread('fognoise.png', cv2.IMREAD_COLOR)
    img = histogram(img)
    cv2.imwrite('resultImages/Q1/fogNoiseHistogram.png', img)
    cv2.imshow('Fog And Noise Histogram & cumHisto', img)


def frostFogH():
    img = cv2.imread('frostfog.png', cv2.IMREAD_COLOR)
    img = histogram(img)
    cv2.imwrite('resultImages/Q1/frostFogHistogram.png', img)
    cv2.imshow('Frost fog Histogram & cumHisto', img)


#Q2

def meanAndGaussian(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)

    #Mean Filter
    kernel = np.ones((5, 5), np.float32) / 25
    mean = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('resultImages/Q2/cameramanMeanFilter.png',mean)
    cv2.imshow('cameraman image after applying mean filter', mean)

    meanHist = histogram(mean)
    cv2.imwrite('resultImages/Q2/cameramanMeanFilterHistogram.png', meanHist)

    #Gaussian filter
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('resultImages/Q2/cameramanGaussianFilter.png', gaussian)
    cv2.imshow('cameraman image after applying gaussian filter', gaussian)

    gaussianHist = histogram(gaussian)
    cv2.imwrite('resultImages/Q2/cameramanGaussianFilterHistogram.png', gaussianHist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# def get_median(image, r, c):
#     l = []
#
#     l.append(image[r-2][c-2])
#     l.append(image[r-2][c-1])
#     l.append(image[r-2][c])
#     l.append(image[r-2][c+1])
#     l.append(image[r-2][c+2])
#
#     l.append(image[r-1][c-2])
#     l.append(image[r-1][c-1])
#     l.append(image[r-1][c])
#     l.append(image[r-1][c+1])
#     l.append(image[r-1][c+2])
#
#     l.append(image[r][c - 2])
#     l.append(image[r][c - 1])
#     l.append(image[r][c])
#     l.append(image[r][c + 1])
#     l.append(image[r][c + 2])
#
#     l.append(image[r + 1][c - 2])
#     l.append(image[r + 1][c - 1])
#     l.append(image[r + 1][c])
#     l.append(image[r + 1][c + 1])
#     l.append(image[r + 1][c + 2])
#
#     l.append(image[r + 2][c - 2])
#     l.append(image[r + 2][c - 1])
#     l.append(image[r + 2][c])
#     l.append(image[r + 2][c + 1])
#     l.append(image[r + 2][c + 2])
#
#     l = np.sort(l)
#     return l[12][0]

#Q3

def medianFilter(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    print(img.shape)
    result = np.zeros((960,1280 , 3), dtype=np.int)
    print(result.shape)

    start1 = timeit.default_timer()

    median = cv2.medianBlur(img, 5)
    cv2.imwrite('resultImages/Q3/fogNoiseMedianFilter.png', median)
    cv2.imshow('fog and noise image after applying median filter', median)

    medianHist = histogram(median)
    cv2.imwrite('resultImages/Q3/fogNoiseMedianFilterHistogram.png', medianHist)

    stop1 = timeit.default_timer()
    print(stop1 - start1)


    # filter out of range 31 and 244

    start2 = timeit.default_timer()

    for i in range(3, img.shape[0] -4):
        for j in range(3, img.shape[1]-4):
            if (np.any(img[i, j] <= 31)):
                #result[i, j] = get_median(img,i,j)
                result[i, j] = cv2.medianBlur(img[i,j], 5)
            else:
                if (np.any(img[i, j] >= 244)):
                    #result[i, j] = get_median(img, i, j)
                    result[i, j] = cv2.medianBlur(img[i, j], 5)
                else:
                    result[i,j] = img[i,j]

    cv2.imwrite('resultImages/Q3/fogNoiseRangeMedianFilter.png', result)
    cv2.imshow('fog and noise image after applying median filter on specific range', result)

    medianHist1 = histogram(result)
    cv2.imwrite('resultImages/Q3/fogNoiseRangeMedianFilterHistogram.png', medianHist1)

    stop2 = timeit.default_timer()
    print(stop2 - start2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Q4

def equalization(img):
    img = cv2.imread(img, 0)
    equ = cv2.equalizeHist(img)
    cv2.imwrite('resultImages/Q4/frostFogEqualized.png', equ)
    cv2.imshow('Frost fog image after applying histogram equalization', equ)

    equHist = histogram(equ)
    cv2.imwrite('resultImages/Q4/frostFogEqualizedHistogram.png', equHist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def stretching(img):
    img = cv2.imread(img, 0)

    histogramList = np.zeros(256, dtype=np.int)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            histogramList[img[i, j]] += 1

    c = 0
    d = 0
    for i in range(0,len(histogramList)):
        if(histogramList[i] != 0):
            c = i
            break

    for i in range(len(histogramList) -1 ,-1,-1):
        if(histogramList[i] !=0):
            d = i
            break
    #print(c)
    #print(d)

    a = 0
    b = 255
    scaleFactor = ((b-a)/(d-c))

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i,j] = int((img[i,j] -c ) * scaleFactor + a)

    cv2.imwrite('resultImages/Q4/frostFogStretched.png', img)
    cv2.imshow('Frost fog image after applying Contrast stretching', img)

    strHist = histogram(img)
    cv2.imwrite('resultImages/Q4/frostFogStretchedHistogram.png', strHist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Bonus
def bonus():
    treeM = cv2.imread('treeM.png', 0)
    tree = cv2.imread('tree.png', 0)

    result = (treeM - tree) * 50

    cv2.imwrite('resultImages/bonus/bonus.png', result)
    cv2.imshow('Extracting the mystery', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




#Tests


# cameramanH()
# batmanH()
# fogH()
# fogAndNoiseH()
# frostFogH()
# meanAndGaussian('cameraman.png')
# medianFilter('fognoise.png')
# equalization('frostfog.png')
# stretching('frostfog.png')
# bonus()