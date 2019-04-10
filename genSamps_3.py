import os
import numpy as np
from scipy import misc

divBy = 3000.0
middleNorm = (255.0/divBy)/2

filesInSamps2 = sorted(os.listdir('samps_2'))

def gen(prefix,totalToTake):

    #### LOAD DATA
    xdic = {}
    count = 0

    for f in filesInSamps2:
        if f in ['.DS_Store','.gitignore']:
            continue
        myImage = misc.imread('samps_2/'+f)
        if f[0] == prefix:
            count+=1
            xSize = len(myImage.flatten())
            xdic[f] = (myImage.flatten() / divBy) - middleNorm
            if count >= totalToTake:
                break

    x = []
    for k in sorted(xdic.keys()):
        x.append(xdic[k])

    xnp = np.array(x)
    xlen = xnp.shape[0]

    xret = xnp.reshape(xlen,xSize)
    return xret



def revy(inArr,ittrNumb,fileNo,suf,outdir):
    arr = np.copy(inArr)
    arr += middleNorm
    arr *= divBy
    newImage = np.reshape(arr, (33,33,3))
    # misc.imsave('out/'+str(ittrNumb)+suf+".jpg",newImage)
    # print("hello?")
    if "samp" in suf :
        # print(ittrNumb,fileNo,suf,filesInSamps2[fileNo][1:])
        compositeImage = misc.imread('samps_1/'+filesInSamps2[fileNo][1:])
        misc.imsave(outdir+''+str(ittrNumb)+suf+"_orig.jpg",compositeImage)
        compositeImage[33:66,33:66] = newImage
        misc.imsave(outdir+''+str(ittrNumb)+suf+"_comp.jpg",compositeImage)


def revx(inArr,ittrNumb,fileNo,suf,outdir):
    arr = np.copy(inArr)
    arr += middleNorm
    arr *= divBy
    newImage = np.reshape(arr, (33,132,3))
    misc.imsave(outdir+''+str(ittrNumb)+suf+".jpg",newImage)

