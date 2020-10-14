# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 17:12:17 2020

@author: Home-PC
"""

import pandas as pd
import numpy as np
import cv2
import os

fer_data=pd.read_csv('C:/PAC/dissertation/fer2013.csv',delimiter=',')



for index,row in fer_data.iterrows():
    pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
    img=pixels.reshape((48,48))
    pathname=os.path.join('fer_images',str(index)+'.jpg')
    cv2.imwrite(pathname,img)
    print('image saved ias {}'.format(pathname))