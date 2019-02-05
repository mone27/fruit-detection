import cv2
import numpy as np

# Copyright


class ClassicSegmentation:
    """A class for image segmentation usage:"""

    def __init__(self, **kwargs):
        """parameter:
            filename (must be specified) path to the image file
            threshold = 100
            denoise = 3
            min_area = 3
            max_area = 50
            """
        self.cut_left = kwargs.get('cut_left', 5)
        self.cut_right = kwargs.get('cut_right', 95)
        self.cut_top = kwargs.get('cut_top', 5)
        self.cut_bottom = kwargs.get('cut_bottom', 95)
        self.image = cv2.imread(str(kwargs['filename']))
        self.mask = None
        self.denoise_size = kwargs.get('denoise', 3)
        self.threshold = kwargs.get('threshold', 100)
        width = self.image.shape[:2][1]
        self.min_area = int(width * kwargs.get('max_area', 3) / 100) ** 2  # set default value
        self.max_area = int(width * kwargs.get('min_area', 50) / 100) ** 2  # set default value

        self.individual_masks = []
        self.individual_images = []
        # can call get_mask() here...

    def get_mask(self):
        self.mask = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.mask = cv2.threshold(self.mask, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._floodfill()._denoise()
        self._clean_mask()

    def _cut_image(self, cut_left, cut_right, cut_top, cut_bottom):
        h, w = self.image.shape[:2]
        w_left = int(w * self.cut_left / 100)
        w_right = int(w * self.cut_right / 100)
        h_top = int(h * self.cut_top / 100)
        h_bottom = int(h * self.cut_bottom / 100)
        self.image = self.image[h_top:h_bottom, w_left:w_right, :]

    def _floodfill(self):
        mask_floodfill = self.mask.copy()
        hf, wf = self.mask.shape[:2]
        mask = np.zeros((hf + 2, wf + 2), np.uint8)
        cv2.floodFill(mask_floodfill, mask, (0, 0), 255)
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        self.mask = self.mask | mask_floodfill_inv
        return self

    def show(self):
        # can refactor to use
        cv2.namedWindow("image: ", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask: ", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("mask: ", 1920, 1080)
        cv2.imshow("image: ", self.image)
        cv2.imshow("mask: ", self.mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _denoise(self):
        kernel = np.ones((self.denoise_size, self.denoise_size), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel, iterations=2)
        return self

    @staticmethod
    def get_contours(image):
        """just a wrapper around cv2.findContours"""
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _clean_mask(self):
        contours = self.get_contours(self.mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.mask, (x, y), (x + w + 10, y + h + 10), 0, -1)

    def _clean_image(self):
        self.image = self.image & \
                np.   stack([self.mask, self.mask, self.mask], axis=2) #to make background perfectly black

    def _calculate_individual_masks(self):
        if self.individual_masks:  # skip if already executed
            return
        self.get_mask()
        self._clean_image()
        contours = self.get_contours(self.mask)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            image_section = self.image[y:y + h, x:x + w]
            mask_section = self.mask[y:y + h, x:x + w]
            self.individual_masks.append(image_section)
            self.individual_images.append(mask_section)

    @property
    def small_masks(self):
        self._calculate_individual_masks()
        return self.individual_masks

    @property
    def small_images(self):
        self._calculate_individual_masks()
        return self.individual_images


mask = ClassicSegmentation(filename="Dataset/Albicocca  /albicocche1.tif")
