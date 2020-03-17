from pathlib import Path
from collections import namedtuple
from functools import lru_cache
from itertools import product
from fastai.vision import *
from math import ceil

# not so nice hardcoded constant
CACHE_MAX_SIZE = 20

# looks like a non consistent API to have mask in ImageTile
ImageTile = namedtuple('ImageTile', 'path idx rows cols custom_bg mask')
ImageMaskTile = namedtuple('ImageMaskTile', 'path idx cols rows')


class CustomBackground:
    def __init__(self, bg_path: Image, tfms: TfmList = []):
        """bgs_dir a folder with all backgrounds, tfms list of transformation to be applied on bgs"""
        self.bg = open_image(bg_path)
        self.tfms = tfms

    def apply(self, img: Image, mask: ImageSegment) -> Image:
        """replaces the background in"""
        mask = mask.data.type(torch.ByteTensor)
        new_bg = self.repeat_bg_tfms(img.size)
        return Image(torch.where(mask, img.data, new_bg.data))

    def repeat_bg_tfms(self, size: Tuple[int, int]) -> Image:
        y, x = size
        bg_z, bg_y, bg_x = self.bg.shape
        nx = ceil(x / bg_x)
        ny = ceil(y / bg_y)
        f_bg = torch.empty((bg_z, ny * bg_y, nx * bg_x), dtype=self.bg.data.dtype)
        for ix, iy in product(range(nx), range(ny)):
            f_bg[:, bg_y * iy:bg_y * (iy + 1), bg_x * ix:bg_x * (ix + 1)] = self.bg.apply_tfms(self.tfms).data
        return Image(f_bg[:, :y, :x])


@lru_cache(maxsize=CACHE_MAX_SIZE)
def open_image_cached(*args, **kwargs): return open_image(*args, **kwargs)


@lru_cache(maxsize=CACHE_MAX_SIZE)
def open_mask_cached(*args, **kwargs): return open_mask(*args, **kwargs)


def get_image_tile(img: Image, idx, rows, cols) -> Image:
    row = idx // cols
    col = idx % cols
    tile_x = img.size[1] // cols
    tile_y = img.size[0] // rows
    # need to use .px instead of .data because ImageSegment convert data to int64
    return Image(img.px[:, row * tile_y:(row + 1) * tile_y, col * tile_x:(col + 1) * tile_x])


def open_image_tile(img_t: ImageTile, **kwargs) -> Image:
    """given and ImageTile it returns and Image with the tile,
    set mask to True for masks"""
    path, idx, rows, cols, bg, mask_t = img_t
    img = open_image_cached(path, **kwargs)
    img = get_image_tile(img, idx, rows, cols)

    if bg:
        mask = open_mask_tile(mask_t)
        img = bg.apply(img, mask)
    return img


def open_mask_tile(mask_t: ImageMaskTile, **kwargs):
    path, idx, rows, cols = mask_t
    img = open_mask_cached(path, **kwargs)
    return ImageSegment(get_image_tile(img, idx, rows, cols).px)


def get_tiles(images, rows: int, cols: int) -> Collection[ImageTile]:
    images_tiles = []
    for img in images:
        for i in range(rows * cols):
            images_tiles.append(ImageTile(img, i, rows, cols, None, None))
    return images_tiles


def add_bg_to_tiles(img_tiles, custom_bgs, get_mask_fn):
    img_bg_tiles = []
    for bg in custom_bgs:
        for tile in img_tiles:
            img_bg_tiles.append(ImageTile(*tile[:4], bg, get_mask_fn(tile)))
    return img_bg_tiles

class SegmentationTileLabelList(SegmentationLabelList):
    # hardcoded div=True
    def open(self, fn: ImageTile):
        return open_mask_tile(fn, div=True)


class SegmentationTileItemList(ImageList):
    _label_cls, _square_show_res = SegmentationTileLabelList, False

    # accepts as inputs a tuple of TileInfo and PathOrStr
    def open(self, fn: ImageTile) -> Image:
        return open_image_tile(fn, convert_mode=self.convert_mode, after_open=self.after_open)

    @classmethod
    def from_folder(cls, path, rows, cols, custom_bgs=None, get_mask_fn=None, **kwargs) -> ItemList:
        """Create an `ItemList` in `path` from the filenames that have a suffix in `extensions`.
        `recurse` determines if we search subfolders."""
        files = get_files(path, recurse=True)
        img_tiled = get_tiles(files, rows, cols)
        if custom_bgs and get_mask_fn:
            img_tiled = add_bg_to_tiles(img_tiled, custom_bgs, get_mask_fn)
        return SegmentationTileItemList(img_tiled, **kwargs)

def seg_accuracy(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()


class SemanticSegmentationTile:
    """Class for all semantic segmentation"""

    codes = ['background', 'fruit']

    def __init__(self, path_imgs: Path, path_masks: Path, max_tile_size=512, bs=16, max_tol=30,
                 transforms=get_transforms(), model=models.resnet34, custom_bg_dir=None, bg_tfms=[]):
        """
        images and masks must be all of the same size!
        """
        self.path_imgs = path_imgs
        self.path_msks = path_masks

        bgs = []
        if custom_bg_dir:
            for bg in get_files(custom_bg_dir):
                bgs.append(CustomBackground(bg, bg_tfms))

        # need to get 1 image find the size and call calc_n_tile        
        self.img_size = open_image(get_files(path_imgs, None, recurse=True)[0]).size

        (self.rows, self.y_tile, self.y_diff), (self.cols, self.x_tile, self.x_diff) \
            = SemanticSegmentationTile.calc_n_tiles(self.img_size, max_tile_size, max_tol)
        # Maybe need to use Logger instead of print
        print(f"Creating Dataset of images of total size: ({self.img_size[0]}, {self.img_size[1]});"
              f"\nnumber of rows: {self.rows}, columns: {self.cols};"
              f"\nsize of tiles: ({self.y_tile}, {self.x_tile});"
              f"\ndiscared pixels due to rounding y: {self.y_diff} x: {self.x_diff}")
        self.data = (SegmentationTileItemList
                     .from_folder(self.path_imgs, self.rows, self.cols, bgs, self.get_mask_tiles)
                     .split_by_rand_pct()
                     .label_from_func(self.get_mask_tiles, classes=self.codes)
                     .transform(transforms ,tfm_y=True)
                     .databunch(bs=bs)
                     .normalize(imagenet_stats)) # needed because it use pretrained resnet on ImageNet
        print(f"created Dataset with in: {len(self.data.train_ds)} tiles train and {len(self.data.valid_ds)} in valid",
             "\ncreating learner ...")
        self.learn = unet_learner(self.data, model, metrics=seg_accuracy)
        print("done")
        

    def get_maskp(self, path):
        return self.path_msks / path.name

    def get_mask_tiles(self, image_tile: ImageTile):
        path, idx, rows, cols, _, _ = image_tile
        path = self.get_maskp(path)
        return ImageMaskTile(path, idx, rows, cols)


    @staticmethod
    def best_block_divide(sz, max_blk_sz, max_tol):
        """sz: size of the original input
            max_blk_sz: max block size (returned block size must be <=)
            max_tol: max number of elements left, expressed in number of elements.

            finds the min number of block that satisfies this conditions
            returns tuple of:
            n_blocks, block_size, n_elem_left
            """
        blk_sz = max_blk_sz
        # migrate to walrus operator in py 3.8
        diff = (sz % blk_sz)
        # try with smaller blk_size until the diff is <= max_diff
        # edge case is blk_size=1 which means diff==0
        while diff > max_tol:
            blk_sz -= 1
            diff = (sz % blk_sz)
        return sz // blk_sz, blk_sz, sz % blk_sz

    @staticmethod
    def calc_n_tiles(size, tile_max_size, max_tol=30):
        """returns the \"best\" size for the tile given the image size.
        return type is a tuple of (n_blocks, block_size, n_elem_left)"""
        y, x = size
        x = SemanticSegmentationTile.best_block_divide(x, tile_max_size, max_tol)
        y = SemanticSegmentationTile.best_block_divide(y, tile_max_size, max_tol)
        return y, x

    def predict_mask(self, img_path):
        # init mask, it is set to 0 because it is bigger than the sum of all tiles
        mask = torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.int64)
        # Maybe need to use pred_batch for better performance
        for row, col in product(range(self.rows), range(self.cols)):
            tile_idx = row * self.cols + col  # get the index of the tile
            img_tile = ImageTile(img_path, idx=tile_idx, rows=self.rows, cols=self.cols, custom_bg=None, mask=None)
            img_tile = open_image_tile(img_tile)
            mask_tile, _, _ = self.learn.predict(img_tile)

            mask[:, self.y_tile * row:self.y_tile * (row + 1),
            self.x_tile * col:self.x_tile * (col + 1)] = mask_tile.data
        return ImageSegment(mask)

    def seg_test_image_tile(self, img: ImageTile, real_mask: ImageTile):
        pred_mask, _, _ = inf_learn.predict(open_image_tile(img))
        real_mask = open_image_tile(real_mask, div=True, mask=True)
        wrong = np.count_nonzero(pred_mask.data != real_mask.data)
        intersection = np.logical_and(real_mask.data, pred_mask.data)
        union = np.logical_or(real_mask.data, pred_mask.data)
        iou_score = torch.div(torch.sum(intersection), torch.sum(union).float())

        return iou_score, wrong

    # warning img_size: number of pixels
    def seg_benchmark_tile(self, path_img):
        "benchmarks the full dataset in path_img analysing the single tiles"
        imgs = get_image_files(path_img)
        imgs = get_tiles(imgs, rows, cols)
        masks = [get_labels_tiles(f) for f in imgs]
        wrongs = torch.empty(len(imgs), dtype=torch.float32)
        iou_scores = torch.empty(len(imgs), dtype=torch.float32)
        for i, img, mask in zip(range(len(imgs)), imgs, masks):
            iou_score, wrong = seg_test_image_tile(img, mask)
            wrongs[i] = wrong
            iou_scores[i] = iou_score

        wrong_perc = torch.mean(wrongs) * 100 / imgs[0].size.prod() # should be image size
        print(f"perc of wrong pixels: {wrong_perc}%")
        print(f"mean accuracy: {100 - wrong_perc}%")
        print(f"max wrong per image: {torch.max(wrongs)} over {img_size} pixels")
        print(f"mean IoU: {torch.mean(iou_scores)}")
        print(f"min IoU: {torch.min(iou_scores)}")

    def seg_test_full_image(self, img_p: PathOrStr):
        pred_mask = predict_mask(img_p)
        real_mask = open_mask(get_label(img_p), div=True)
        pred_mask, real_mask = pred_mask.data, real_mask.data
        wrong = np.count_nonzero(pred_mask != real_mask)

        intersection = np.logical_and(real_mask, pred_mask)
        union = np.logical_or(real_mask, pred_mask)
        iou_score = torch.div(torch.sum(intersection), torch.sum(union).float())

        return iou_score, wrong

    def plot_pixel_difference_tile(img, real_mask, figsize=(20, 20)):
        img = open_image_tile(img)
        pred_mask, _, _ = inf_learn.predict(img)
        real_mask = open_image_tile(real_mask, mask=True, div=True)
        real_mask = image2np(real_mask.data)
        pred_mask = image2np(pred_mask.data)
        img = image2np(img.data)
        diff = pred_mask != real_mask
        img[diff == 1] = (1, 0, 0)  # 0000FF is blue FF0000 is red
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

    def plot_pixel_difference_full(img_p, mask_p, figsize=(20, 20)):
        img = open_image(img_p)
        pred_mask = predict_mask(img_p)
        real_mask = open_mask(mask_p, div=True)
        real_mask = image2np(real_mask.data)
        pred_mask = image2np(pred_mask.data)
        img = image2np(img.data)
        diff = pred_mask != real_mask
        #     breakpoint()
        img[diff == 1] = (1, 0, 0)  # 0000FF is blue FF0000 is red
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)


# %% tests

if __name__ == '__main__':
    segm = SemanticSegmentationTile(Path("dataset_segmentation/images"), Path("dataset_segmentation/labels"),
                                    max_tile_size=256, model=models.resnet18,
                                    custom_bg_dir="backgrounds", bg_tfms=get_transforms()[0])

    segm.learn = segm.learn.load("resnet18_4_epoch_freezed_1e-4_bg_0000")
    print("predicting mask")
    # mask1 = segm.learn.predict(open_image("dataset_segmentation/images/albicocche1.png"))
    mask, _, _ = segm.learn.predict(segm.data.valid_ds[0][0])
    mask.show()
