import cv2
import csv
import numpy as np
from os import listdir, makedirs, path
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import pylab
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
pylab.rcParams['figure.figsize'] = (100, 50)
args =  {'dir_selection': 'Cetriolo', 'file_selection': 'cetrioli4.tif', 'cut_left': 5, 'cut_right': 95, 'cut_top': 5, 'cut_bottom': 95, 'method': 'BINARY+OTSU', 'min_area': 3, 'max_area': 50, 'threshold': 100, 'denoise': 3, 'blocksize': 101, 'plot': False, 'plot_extracted': False, 'save_files': True}
trf = 'Scan'  # subfolder scans
root_folder = Path('/home/simone/development/python_università/Mario/RAW DATA')
ext = '.tif'  # source file extention
ext_save = '.tif'  # save file extention
root_folder_save = Path('/home/simone/development/python_università/Dataset_extracted_images')


def segmentation_func(dir_selection, file_selection, cut_left, cut_right, cut_top, cut_bottom, method, min_area, max_area, threshold, denoise, blocksize, plot, plot_extracted, save_files):
    print(f"processing: {locals()}")
    # must split this function in sub function
    fullname_fg = root_folder / dir_selection / file_selection #join paths using pathlib to abstract OS differences
    img = cv2.imread(str(fullname_fg))
    img_cut = img.copy()
    h, w = img_cut.shape[:2]
    max_area = int(w * max_area / 100) ** 2
    min_area = int(w * min_area / 100) ** 2
    print(f"min area: {min_area}, max area: {max_area}")
    w_left = int(w*cut_left/100)
    cv2.line(img_cut,(w_left, 0),(w_left, h),(0,255,0), int(w*0.01))
    w_right = int(w*cut_right/100)
    cv2.line(img_cut,(w_right, 0),(w_right, h),(255,255,0), int(w*0.01))
    h_top = int(h*cut_top/100)
    cv2.line(img_cut,(0, h_top),(w, h_top),(0,0,255), int(w*0.01))
    h_bottom = int(h*cut_bottom/100)
    cv2.line(img_cut,(0, h_bottom),(w, h_bottom),(255,0,255), int(w*0.01))
    img = img[h_top:h_bottom, w_left:w_right,:]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if method == 'BINARY':
        ret, th1 = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    if method == 'BINARY+OTSU':
        ret, th1 = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if method == 'ADAPTIVE_MEAN':
        th1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, 0)
    if method == 'ADAPTIVE_GAUSSIAN':
        gray = cv2.GaussianBlur(gray,(5,5),0)
        th1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 0)

    #floodfill
    th1_floodfill = th1.copy()
    hf, wf = th1.shape[:2]
    mask = np.zeros((hf + 2, wf + 2), np.uint8)
    cv2.floodFill(th1_floodfill, mask, (0, 0), 255)
    th1_floodfill_inv = cv2.bitwise_not(th1_floodfill)
    th1 = th1 | th1_floodfill_inv


    #denoise
    kernel = np.ones((denoise,denoise),np.uint8)
    th1 = cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel, iterations = 2)



    _, contours, hi = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get countours
    #writer countors to image
    img_contours = img.copy()

    cropped_images = []
    n_samples = 0
    for i, cnt in enumerate(contours):
        X, Y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if (area < min_area or area > max_area):
            continue

        section = img[Y:Y + h, X:X + w]
        mask = th1[Y:Y + h, X:X + w]
        cv2.drawContours(img_contours, contours, i, (0, 255, 0), 3)
        cropped_images.append((section, mask))
        n_samples += 1


    if plot == True:
        images = [img_cut[:,:,::-1], th1, img_contours[:,:,::-1]]
        colors = [None, 'gray', None]
        if plot_extracted:
            n_rows = (len(cropped_images) // 3) + 3 # to ensure enogh space for cropped images plus orginal images
            for idx in range(len(cropped_images)):
                plt.subplot(n_rows ,3,3 + idx + 1),plt.imshow(cropped_images[idx][0][:,:,::-1])
        else:
            for i in range(3):
                plt.subplot(1,3,i+1),plt.imshow(images[i], colors[i])
        plt.show()

    print("Detected samples:", str(n_samples))
    if save_files:
        folder_save = root_folder_save / dir_selection
        print(f"saving to files to {folder_save}")
       # makedirs(folder_save)
        for i in range(len(cropped_images)):
            image, mask = cropped_images[i]
            #dirty image
            filename_image_dirty = folder_save / f"{file_selection[:-len(ext)]}_{i}_dirty_{ext_save}" #generate file name
            cv2.imwrite(str(filename_image_dirty), image) #images
            # mask
            filename_mask = folder_save / f"{file_selection[:-len(ext)]}_{i}_mask_{ext_save}"
            cv2.imwrite(str(filename_mask), mask)
            # inverted mask
            filename_mask = folder_save / f"{file_selection[:-len(ext)]}_{i}_mask_inv{ext_save}"
            cv2.imwrite(str(filename_mask), cv2.bitwise_not(mask))
            # clean image
            filename_image_clean = folder_save / f"{file_selection[:-len(ext)]}_{i}_clean_{ext_save}"
            print(f"{mask.shape} and  {image.shape}")
            # mask_3d = np.empty(image.shape)
            # mask_3d[:]
            image_clean = image & \
                np.stack([mask, mask, mask], axis=2) #to make background perfectly black
            cv2.imwrite(str(filename_image_clean), image_clean)
            print(f"saved file: {filename_image_dirty}")

segmentation_func(**args)
