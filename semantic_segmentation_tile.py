from pathlib import Path
from collections import namedtuple
from functools import lru_cache
from fastai.vision import *

# not so nice hardcoded costant
CACHE_MAX_SIZE = 20


ImageTile = namedtuple('ImageTile', 'path idx rows cols')

@lru_cache(maxsize=CACHE_MAX_SIZE)
def open_image_cached(*args, **kwargs): return open_image(*args, **kwargs)

@lru_cache(maxsize=CACHE_MAX_SIZE)
def open_mask_cached(*args, **kwargs): return open_mask(*args, **kwargs)

def open_image_tile(img_t: ImageTile, mask=False, **kwargs) -> Image:
    """given and ImageTile it returns and Image with the tile,
    set mask to True for masks, indexes ... TODO"""
#     print(f"opening image Tile {img_t}, mask: {mask}")
    path, idx, rows, cols = img_t
    img = open_image_cached(path ,**kwargs) if not mask else open_mask_cached(path, **kwargs)

    row = idx // cols
    col = idx % cols
    tile_x = img.size[1] // cols
    tile_y = img.size[0] // rows
    return Image(img.data[:,  row * tile_y:(row + 1) * tile_y , col * tile_x:(col + 1) * tile_x])

def get_tiles(images: PathOrStr, rows: int, cols: int) -> Collection[ImageTile]:
    images_tiles = []
    for img in images:
        for i in range(rows * cols):
            images_tiles.append(ImageTile(img, i, rows, cols))
    return images_tiles

class SegmentationTileLabelList(SegmentationLabelList):

    def open(self, fn: ImageTile):
        return open_image_tile(fn, div=True, mask=True)


class SegmentationTileItemList(ImageList):
    _label_cls, _square_show_res = SegmentationTileLabelList, False

    # accepts as inputs a tuple of TileInfo and PathOrStr
    def open(self, fn: ImageTile) -> Image:
        return open_image_tile(fn, convert_mode=self.convert_mode, after_open=self.after_open)

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', rows=1, cols=1, extensions: Collection[str] = None, **kwargs) -> ItemList:
        """Create an `ItemList` in `path` from the filenames that have a suffix in `extensions`.
        `recurse` determines if we search subfolders."""
        files = get_files(path, extensions, recurse=True)
        files_tiled = get_tiles(files, rows, cols)
        return SegmentationTileItemList(files_tiled, **kwargs)

def seg_accuracy(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

class SemanticSegmentationTile:
    
    """Class for all semantic segmentation"""

    codes = ['background', 'fruit']

    def __init__(self, path_imgs: Path, path_masks: Path, max_tile_size=512, bs=16, max_tol=30, transforms=get_transforms()):
        """
        images and masks must be all of the same size!
        """
        self.path_imgs = path_imgs
        self.path_msks = path_masks
        
        # need to get 1 image find the size and call calc_n_tile        
        img_size = open_image(get_files(path_imgs, None, recurse=True)[0]).size
        
        (self.rows, self.x_tile, self.x_diff), (self.cols, self.y_tile, self.y_diff)\
            = SemanticSegmentationTile.calc_n_tiles(img_size, max_tile_size, max_tol)
        # Maybe need to use Logger instead of print
        print(f"Creating Dataset of images of total size: ({img_size[0]}, {img_size[1]});"
              f"\nnumber of rows: {self.rows}, columns: {self.cols};"
              f"\nsize of tiles: ({self.y_tile}, {self.x_tile});"
              f"\ndiscared pixels due to rounding y: {self.y_diff} x: {self.x_diff}")
        self.data = (SegmentationTileItemList
                .from_folder(self.path_imgs, self.rows, self.cols)
                .split_by_rand_pct()
                .label_from_func(self.get_mask_tiles, classes=self.codes)
                .transform(transforms ,tfm_y=True)
                .databunch(bs=bs)
                .normalize(imagenet_stats)) # needed because it use pretrained resnet on ImageNet
        print(f"created Dataset with in: {len(self.data.train_ds)} tiles train and {len(self.data.valid_ds)} in valid",
             "\ncreating learner ...")
        self.learn = unet_learner(self.data, models.resnet34, metrics=seg_accuracy)
        print("done")
        

    def get_maskp(self, path):
        return self.path_msks / path.name

    def get_mask_tiles(self, image_tile: ImageTile):
        path, *tile = image_tile
        path = self.get_maskp(path)
        return ImageTile(path, *tile)

    # Need to find a better name!
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
        x, y = size
        x = SemanticSegmentationTile.best_block_divide(x, tile_max_size, max_tol)
        y = SemanticSegmentationTile.best_block_divide(y, tile_max_size, max_tol)
        return x, y

    #TODO refactor to use new image notation
    def predict_mask(self, img_path):
        # uses all the global variables that are output of calc_n_tiles(), maybe refactor
        i = 0
        end_mask = None  # sentinel value for first iteration
        for col in range(cols):
            col_mask = None
            for row in range(rows):
                img_tile = open_image_tile(ImageTile(img_path, idx=i, rows=rows, cols=cols))
                mask_tile, _, _ = inf_learn.predict(img_tile)
                mask_tile = mask_tile.data.permute(1, 2,
                                                   0)  # convert to numpy style it is the way to think about the image
                col_mask = np.vstack((col_mask, mask_tile)) if col_mask is not None else mask_tile
                i += 1
            end_mask = np.hstack((end_mask, col_mask)) if end_mask is not None else col_mask
        # add padding to make pred_mask size the same of the original image
        pred_mask = torch.zeros((*img_size, 1), dtype=torch.long)  # iniziatlize mask with correct size with all black
        pred_mask[:-x_diff, :-y_diff, :] = torch.tensor(end_mask).permute(1, 0,
                                                                          2)  # copy mask on the all black background
        return Image(pred_mask.permute(2, 1, 0))  # convert back to pytorch image style

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

# %% testssss

if __name__=='__main__':
    trs = get_transforms()
    def get_maskp(path):
        return Path("dataset_segmentation/labels") / path.name

    def get_mask_tiles(image_tile: ImageTile):
        path, *tile = image_tile
        path = get_maskp(path)
        return ImageTile(path, *tile)

    src = (SegmentationTileItemList
                .from_folder("dataset_segmentation/images", 2,2)
                .split_none()
                .label_from_func(get_mask_tiles, classes=['background', 'fruit']))


    src2 = (SegmentationItemList.from_folder("dataset_segmentation/images")
            .split_none()
            .label_from_func(get_maskp, classes=['background', 'fruit']))

    data=        (src.transform(trs, tfm_y=True)
                .databunch(bs=1)
                .normalize(imagenet_stats))
    # seg = SemanticSegmentationTile(Path("dataset_segmentation/images"), Path("dataset_segmentation/labels"), transforms=trs)