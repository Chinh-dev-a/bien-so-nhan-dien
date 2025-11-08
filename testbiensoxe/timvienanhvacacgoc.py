import cv2
import imutils
import numpy as np
import matplotlib as plt
import os

from Tools.scripts.highlight import ansi_highlight
from networkx import periphery
from networkx.algorithms.bipartite.cluster import cc_max
from numpy.distutils.lib2def import output_def
from sympy.categories import xypic_draw_diagram
from tensorflow.python.framework.ops import convert_n_to_tensor
from tensorflow.python.keras.backend import epsilon


# from test.testnhandienbiensoxe import approx


# def contour()

def main():
    img=cv2.imread('biensoxe/22_plate1.jpg')
    img=cv2.resize(img,(300,175),interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((5, 5), np.uint8)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#chuyen anh sang xam
    cv2.imshow('anh xam', gray)
    gray_denoised = cv2.bilateralFilter(gray, 7, 75, 75)#giam nhieu
    cv2.imshow('anh xam da giam nhieu', gray)
    _,thersh=cv2.threshold(gray_denoised,150,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('chuyen anh sang anh nhi phan', thersh)
    thersh = cv2.dilate(thersh,kernel,iterations = 0)
    thersh = cv2.morphologyEx(thersh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('chuyen anh sang anh nhi phan', thersh)
    # thersh = cv2.morphologyEx(thersh, cv2.MORPH_GRADIENT, kernel)
    thersh = cv2.bitwise_not(thersh)
    # edges = cv2.Canny(img, 100, 200)
    # cv2.imshow('image', edges)
    cv2.imshow('chuyen anh sang anh nhi phan', thersh)

    # cnt=cv2.findContours(thersh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print(cnt)
    #
    # cnt=imutils.grab_contours(cnt)
    # c=max(cnt,key=cv2.contourArea)
    # output=img.copy()
    # cv2.drawContours(output,[c],-1,(0,255,0),2)
    # (x,y,w,h)=cv2.boundingRect(c)
    # text= "N={}".format(len(c))
    # cv2.putText(output,text,(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,
    #          0.7,(0,255,0),1)
    # cv2.imshow('test ', output)

    #=====ve contour len anh =====#
    # edges = cv2.Canny(thersh, 100, 200)
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    #=====ve contour 4 diemlen anh=====#
    eps=0.5
    # for eps in np.linspace(0.001,0.05,10):
    #     peri=cv2.arcLength(c,True)
    #     approx=cv2.approxPolyDP(c,eps*peri,True)
    #     output=img.copy()
    #     cv2.drawContours(output,[approx],-1,(0,255.0),2)
    #     text="eps={:.4f}, N={}".format(eps,len(approx))
    #     cv2.putText(output,text,(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    #     # print("thong tin : {}".format(text))
    #     cv2.imshow('testelp',output)
    #     cv2.waitKey(0)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thersh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print(M)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    epsilon = 1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print("epsilon =",epsilon)
    cv2.putText(img,approx,)


    cv2.imshow('anh cortour',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()