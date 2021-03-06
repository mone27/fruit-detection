import cv2
import numpy as np
from fastai.vision import Image, ImageSegment, image2np, PathOrStr
from typing import *
import matplotlib.pyplot as plt

OpenCvImage = np.ndarray


def get_contours(image):
    """just a wrapper around cv2.findContours"""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def clean_image(mask, min_area=.5, max_area=50):
    """remove (fills with black) from mask all contours that are smaller or bigger than the given area limits
    expressed in percentage of total area"""
    img_area = np.prod(mask.shape[:2])
    max_area = max_area / 100 * img_area
    min_area = min_area / 100 * img_area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            x, y, w, h = cv2.boundingRect(contour)
            mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
    return mask

class ImageDivider:
    # Attention: this class uses OpenCv images notation (width, height) so means stuff need to indexed (cols, rows)
    def __init__(self, img: OpenCvImage, mask: OpenCvImage, clean_mask=False, min_area=.5, max_area=50):
        """clean mask removes all the object that are smaller or bigger that max/min area expressed in percentage"""
        self.img = img.copy()
        self.mask = mask.copy()
        
        if clean_mask: self.mask = clean_image(self.mask, min_area, max_area)
        self._clean_image()
        self._get_bounding_rects()

    @classmethod
    def from_fastai(cls, img: Image, mask: ImageSegment, **kwargs):
        # need to convert to uint 8 and multiply by 255 to get back to opencv image representation
        return cls(image2np(img.data * 255).astype(np.uint8), image2np(mask.data * 255).astype(np.uint8), **kwargs)

    def images(self) -> List[OpenCvImage]:        
        return self._get_individual_images(self.img)
        
    def masks(self) -> List[OpenCvImage]:
        return self._get_individual_images(self.mask)



    def _get_bounding_rects(self):
        self.bounding_rects = [cv2.boundingRect(contour) for contour in get_contours(self.mask)]

    def _get_individual_images(self, img):
        ret = []
        for x, y, w, h in self.bounding_rects:
            ret.append(self.img[y:y + h, x:x + w])
        return ret

    def _clean_image(self):
        self.img = self.img & self.mask[..., None]



def perc(a, b): return int(a * b / 100)


class ClassicSegmentation:
    """A class for image segmentation usage:"""

    def __init__(self, img: OpenCvImage, threshold=100, cut=(5, 95, 5, 95), denoise_size=3, floodfill=True, clean_mask=True, min_area=.5, max_area=50):
        self.img = img.copy()
        self.denoise_size = denoise_size
        self.threshold = threshold
        self.do_floodfill = floodfill
        self.do_clean = clean_mask
        self.min_area, self.max_area = min_area, max_area
        self.orig_size = self.img.shape[:2]
        w, h = self.orig_size
        self.cut = perc(w, cut[0]), perc(w, cut[1]), perc(h, cut[2]), perc(h, cut[3])

    def get_mask(self):
        self._cut_image()

        self.mask = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, self.mask = cv2.threshold(self.mask, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.do_floodfill: self._floodfill()
        self._denoise()
        if self.do_clean: self.mask = clean_image(self.mask, self.min_area, self.max_area)
        self._add_padding()
        return self.mask

    def _denoise(self):
        kernel = np.ones((self.denoise_size, self.denoise_size), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=2)

    def _add_padding(self):
        mask = np.zeros(self.orig_size, np.uint8)
        mask[self.cut[0]:self.cut[1], self.cut[2]:self.cut[3]] = self.mask
        self.mask = mask

    def _cut_image(self):
        self.img = self.img[self.cut[0]:self.cut[1], self.cut[2]:self.cut[3], :]

    def _floodfill(self):
        mask_floodfill = self.mask.copy()
        hf, wf = self.mask.shape[:2]
        mask = np.zeros((hf + 2, wf + 2), np.uint8)
        cv2.floodFill(mask_floodfill, mask, (0, 0), 255)
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        self.mask = self.mask | mask_floodfill_inv

    def show(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BRG2RGB))
        ax[1].imshow(self.mask)

def open_cv_image(img_path: PathOrStr)-> OpenCvImage:
    img = cv2.imread(str(img_path)) 
    if img is None: raise Exception("Cannot open", img_path)
    return img

def save_cv_image(img_path: PathOrStr,image: OpenCvImage ):
    if cv2.imwrite(str(img_path),image) is None: raise Exception("Cannot open", img_path)

def classic_segmenter(img_path, **kwargs)-> List[OpenCvImage]:
    img = open_cv_image(img_path)
    mask = ClassicSegmentation(img, **kwargs).get_mask()
    return ImageDivider(img, mask).images()

# %% test
if __name__ == "__main__":
    imgcv = cv2.imread("dataset_segmentation/images/Apricot1.png")
    maskcv = cv2.cvtColor(cv2.imread('dataset_segmentation/labels/Apricot1.png'), cv2.COLOR_BGR2GRAY)
    imgs = ImageDivider(imgcv, maskcv, clean_mask=False).images()
