import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, euler_number


def get_priz(region):
    area=region.area/region.image.size
    perimeter=region.perimeter/region.image.size
    ecc=2*region.eccentricity
    holes=euler_number(region.image,2)
    area_filled=(region.area_filled/region.image.size - area)
    cy,cx=region.centroid_local
    cx=cx/region.image.shape[1]
    cy=cy/region.image.shape[0]
    tophole=euler_number(region.image[0:int(region.image.shape[0]/2),:],2)
    conv=region.convex_area/region.image.size
    ver = np.sum(np.mean(region.image, 0)>=0.9)>2
    hor = np.sum(np.mean(region.image, 1)>=0.9)>2
    centerpixel=region.image[int(region.image.shape[0]/2),int(region.image.shape[1]/2)]
    orient=region.orientation
    coeff=1
    res=[area,perimeter,ecc,holes,area_filled, cx,cy,tophole, hor, ver,conv,orient, centerpixel]
    res=coeff*np.array(res)
    return res
letters = ["si","+","-","A","C","E","F","G",
           "H","I","J","L","N","O","P",
           "R","S","sa","sc","sh","sk",
           "sn","so","sp","sr","ss","st","su",
           "sv","sy","T","U","V","W","Y"]
priz=[]
labels=[]
for i in range(len(letters)):
    for j in range(0,10):
        image=cv2.imread(f"train/{letters[i]}/{str(j)}.png",cv2.IMREAD_GRAYSCALE)
        image[image>0]=1
        labarr=label(image)
        if letters[i]=="si":
            reg=regionprops(labarr)[0]
            priz.append(get_priz(reg))
            labels.append(-1)
            reg=regionprops(labarr)[1]
            priz.append(get_priz(reg))
            labels.append(i)
        else:
            reg=regionprops(labarr)[0]
            priz.append(get_priz(reg))
            labels.append(i)
knn = cv2.ml.KNearest_create()
train = np.array(priz,dtype=np.float32)
train = train.reshape(train.shape[0], -1)
response = np.array(labels,dtype=np.float32).reshape(-1,1)
print(train.shape)
print(response.shape)
knn.train(train, cv2.ml.ROW_SAMPLE, response)
for i in range(6):
    test_image =cv2.imread(f"{i}.png",cv2.IMREAD_GRAYSCALE)
    test_image[test_image>0]=1
    testlab=label(test_image)
    allregs=regionprops(testlab)
    string=""
    allregs=sorted(allregs, key=lambda x:x.centroid[1])
    prevc=allregs[0]
    isfirst=0
    for testreg in allregs:
        testpriz=get_priz(testreg)
        testpriz=np.array(testpriz,dtype=np.float32).reshape(1,-1)
        ret, results, neighbours, dist = knn.findNearest(testpriz, k=2)
        letter=""
        if ret>=0:
            dist=testreg.bbox[1]-prevc.bbox[3]
            if abs(dist)>=40 and isfirst:
                string+=" "
            prevc=testreg
            letter=letters[int(ret)]
            if letter[0]=="s":
                letter=letter[1]
        string+=letter
        isfirst=1
    print(string)
    plt.imshow(test_image)
    plt.show()
