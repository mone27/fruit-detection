from pathlib import Path
from dataclasses import astuple
from functools import lru_cache
from itertools import product
from fastai.vision import *
from fastai.vision.interpret import SegmentationInterpretation
from math import ceil
import random

# not so nice hardcoded constant
CACHE_MAX_SIZE = 40

class CustomBackground:
    def __init__(self, bg_path , tfms: TfmList = [], size=None):
        """tfms list of transformation to be applied on bgs"""
        self.bg = open_image_cached(bg_path)
        if size: self.bg = self._repeat_bg(size) # same size to have better data augmentation using final size
        self.bg = self.bg.apply_tfms(tfms)

    def apply(self, img: Image, mask: ImageSegment) -> Image:
        """replaces the background in"""
        mask = mask.data.type(torch.ByteTensor)
        new_bg = self._repeat_bg(img.size)
        return Image(torch.where(mask, img.data, new_bg.data))

    def _repeat_bg(self, size: Tuple[int, int]) -> TensorImage:
        if self.bg.size == size: return self.bg
        y, x = size
        bg_z, bg_y, bg_x = self.bg.shape
        nx = ceil(x / bg_x)
        ny = ceil(y / bg_y)
        f_bg = torch.empty((bg_z, ny * bg_y, nx * bg_x), dtype=self.bg.data.dtype)
        for ix, iy in product(range(nx), range(ny)):
            f_bg[:, bg_y * iy:bg_y * (iy + 1), bg_x * ix:bg_x * (ix + 1)] = self.bg.data
        return Image(f_bg[:, :y, :x])


@dataclass(repr=True)
class BaseImageTile:
    path: PathOrStr
    idx: Tuple[int, int]
    tile_sz: Tuple[int, int]
    scale: float = 1

    def __iter__(self):
        yield from astuple(self)


class ImageSegmentTile(BaseImageTile): pass


@dataclass(repr=True)
class ImageTile(BaseImageTile):
    custom_bg: CustomBackground = None
    mask: ImageSegment = None

def scale_img(img, scale): return img.resize((round(img.size[0] / scale), round(img.size[1] / scale)))
        
@lru_cache(maxsize=CACHE_MAX_SIZE)
def open_image_cached(*args, scale=1, **kwargs) -> Image:
    # replacing after_open, remote possibility stuff can blow up here
    kwargs['after_open'] = partial(scale_img, scale=scale)
    return open_image(*args, **kwargs)

@lru_cache(maxsize=CACHE_MAX_SIZE)
def open_mask_cached(*args, scale=1, **kwargs) -> ImageSegment:
    kwargs['after_open'] = partial(scale_img, scale=scale)
    return open_mask(*args, **kwargs)

def get_image_tile(img: Image, idx, tile_sz) -> Image:
    row, col = idx
    tile_y, tile_x = tile_sz
    # need to use .px instead of .data because ImageSegment convert data to int64
    return Image(img.px[:, row * tile_y:(row + 1) * tile_y, col * tile_x:(col + 1) * tile_x])


def open_image_tile(img_t: ImageTile, **kwargs) -> Image:
    """given and ImageTile it returns and Image with the tile,
    set mask to True for masks"""
    path, idx, tile, scale, bg, mask_t = img_t
    img = open_image_cached(path, scale=scale, **kwargs)
    img = get_image_tile(img, idx, tile)

    if bg:
        mask = open_mask_tile(mask_t)
        img = bg.apply(img, mask)
    return img


def open_mask_tile(mask_t: ImageSegmentTile, **kwargs):
    path, idx, tile, scale = mask_t
    img = open_mask_cached(path, scale=scale, **kwargs)
    return ImageSegment(get_image_tile(img, idx, tile).px)


def get_tiles(images, rows: int, cols: int, tile_info: Collection) -> Collection[ImageTile]:
    images_tiles = []
    for img in images:
        for row, col in product(range(rows), range(cols)):
            images_tiles.append(ImageTile(img, (row, col), *tile_info))
    return images_tiles


def add_custom_bg_to_tiles(img_tiles: Collection[ImageTile], custom_bgs: Collection[Tuple[Path, List]], num_bgs,
                           get_mask_fn):
    img_bg_tiles = []
    for _ in range(num_bgs):
        for tile in img_tiles:
            bg = random.choice(custom_bgs)
            tile.custom_bg = CustomBackground(bg[0], bg[1], size=tile.tile_sz)
            tile.mask = get_mask_fn(tile)
            img_bg_tiles.append(tile)
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
    def from_folder(cls, path, rows, cols, tile_info, custom_bgs=None, num_bgs=3, get_mask_fn=None,
                    **kwargs) -> ItemList:
        """Create an `ItemList` in `path` from the filenames that have a suffix in `extensions`.
        `recurse` determines if we search subfolders."""
        files = get_files(path, recurse=True)
        imgs_tiled = get_tiles(files, rows, cols, tile_info)
        if custom_bgs and get_mask_fn:
            img_tiled = add_custom_bg_to_tiles(imgs_tiled, custom_bgs, num_bgs, get_mask_fn)
        return SegmentationTileItemList(imgs_tiled, **kwargs)

# define metrics for segmenation model
def seg_accuracy(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

def iou(*args): return dice(*args, iou=True)

#custom transform
def _add_color(x, min=-.5, max=.5, channel=None):
    """transfomr to add (or remove) a fixed amount(between -1 and 1) of a color for all pixels"""
    if channel is None: channel = np.random.randint(0, x.shape[0] - 1)
    amount = random.uniform(min,max)
    x[channel] = (x[channel] + amount).clamp(0., 1.)
    return x
add_color = TfmPixel(_add_color)

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
    while diff >= max_tol:
        blk_sz -= 1
        diff = (sz % blk_sz)
    return sz // blk_sz, blk_sz, sz % blk_sz


def calc_n_tiles(size, tile_max_size, max_tol=30):
    """returns the \"best\" size for the tile given the image size.
    return type is a tuple of (n_blocks, block_size, n_elem_left)"""
    y, x = size
    x = best_block_divide(x, tile_max_size, max_tol)
    y = best_block_divide(y, tile_max_size, max_tol)
    return y, x

class SemanticSegmentationTile:
    """Class for all semantic segmentation"""

    codes = ['background', 'fruit']

    def __getattr__(self, attr):
        return getattr(self.learn, attr, None)

    def __init__(self, path_imgs: Path, path_masks: Path, max_tile_size=512, bs=16, max_tol=30,
                 transforms=get_transforms(), model=models.resnet18, scale=1, valid_split=.8, custom_bg_dir=None,
                 num_bgs=3, bg_tfms=[]):
        """
        images and masks must be all of the same size!
        """
        self.path_imgs = path_imgs
        self.path_msks = path_masks        

        # need to get 1 image find the size and call calc_n_tile        
        self.img_size = open_image_cached(get_files(path_imgs, None, recurse=True)[0], scale=scale).size
        
        bgs = None
        if custom_bg_dir:
            bgs = [(bg, bg_tfms) for bg in get_files(custom_bg_dir, recurse=True)]

        (self.rows, self.y_tile, self.y_diff), (self.cols, self.x_tile, self.x_diff) \
            = calc_n_tiles(self.img_size, max_tile_size, max_tol)
        print(f"Creating Dataset of images of total size: ({self.img_size[0]}, {self.img_size[1]});"
              f"\nnumber of rows: {self.rows}, columns: {self.cols};"
              f"\nsize of tiles: ({self.y_tile}, {self.x_tile});"
              f"\ndiscared pixels due to rounding y: {self.y_diff} x: {self.x_diff}")
        self.data = (SegmentationTileItemList
                     .from_folder(self.path_imgs, self.rows, self.cols, ((self.y_tile, self.x_tile),
                                                                         scale), bgs, num_bgs, self.get_mask_tiles)
                     .split_by_rand_pct(valid_split)
                     .label_from_func(self.get_mask_tiles, classes=self.codes)
                     .transform(transforms ,tfm_y=True)
                     .databunch(bs=bs)
                     .normalize(imagenet_stats)) # needed because it use pretrained resnet on ImageNet
        print(f"created Dataset with in: {len(self.data.train_ds)} tiles train and {len(self.data.valid_ds)} in valid",
             "\ncreating learner ...")
        self.learn = unet_learner(self.data, model, metrics=[dice, iou, seg_accuracy])
        print("done")
        

    def get_maskp(self, path):
        return self.path_msks / path.name

    def get_mask_tiles(self, image_tile: ImageTile):
        path, idx, tile_sz, scale, _, _ = image_tile
        path = self.get_maskp(path)
        return ImageSegmentTile(path, idx, tile_sz, scale)

    def predict_mask(self, img_path, scale=1):
        # init mask, it is set to 0 because it is bigger than the sum of all tiles
        mask = torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.int64)
        # Maybe need to use pred_batch for better performance
        for row, col in product(range(self.rows), range(self.cols)):
            img_tile = ImageTile(img_path, (row, col), (self.y_tile, self.x_tile), scale)
            img_tile = open_image_tile(img_tile)
            mask_tile, _, _ = self.learn.predict(img_tile)

            mask[:, self.y_tile * row:self.y_tile * (row + 1),
            self.x_tile * col:self.x_tile * (col + 1)] = mask_tile.data
        return ImageSegment(mask)

    def interpet(self):
        return SegmentationTileInterpretation.from_learner(self.learn, tile_size=(self.y_tile, self.x_tile))
    


class SegmentationTileInterpretation(SegmentationInterpretation):
    def __init__(self, *args, tile_size=None):
        super().__init__(*args)
        self.tile_size = tile_size
        self.n = len(self.ds)
        
    @classmethod
    def from_learner(cls, learn: Learner, tile_size, ds_type: DatasetType = DatasetType.Valid, activ: nn.Module = None):
        preds_res = learn.get_preds(ds_type=ds_type, activ=activ, with_loss=True)
        return cls(learn, *preds_res, tile_size=tile_size) 
    
            
    def topk_by_metric(self, metric, k=10, largest=False):
        metric_name = metric.__name__
        if not hasattr(self, metric_name):
            n = len(self.ds)
            res = torch.empty(n)
            prog_bar = progress_bar(range(n))
            prog_bar.comment = f"calc metric: {metric_name}"
            preds = self.preds
            y_true = self.y_true
            for i in prog_bar:
                res[i] = metric(preds[i][None], y_true[i][None])

            setattr(self, metric_name, res.cpu())
        return getattr(self, metric_name).topk(k, largest=largest)
    
    def plot_topk_metric_pixel_diff(self, metric, start=0, end=10, err_color=(1, 0, 0), n_rows=3, figsize=(20, 20)):        
        if metric == 'loss':
            topk = self.top_losses(self.tile_size, end)
            mname = 'loss'
        else:
            topk = self.topk_by_metric(metric, end)
            mname = metric.__name__
        metric_vals = topk.values[start:]
        indices = topk.indices[start:]
        fig, axes = plt.subplots(3, (end-start) // n_rows, figsize=figsize)
        for metric_val, i, ax in zip(metric_vals, indices, axes.flatten()):
            self.plot_pixel_diff(i, descpr=f"{mname}: {metric_val}", ax=ax)
            
    def plot_pixel_diff(self, i, descrp="", ax=plt.plot, err_color=(1, 0, 0), n_rows=3, figsize=(20, 20)):
        pred = self.pred_class[i]
        real = self.ds[i][1]
        img = self.ds[i][0]
        pred = image2np(pred[None])
        real = image2np(real.data)
        diff = pred != real
        img = image2np(img.data)
        img[diff == 1] = err_color
        ax.imshow(img)
        ax.set_title(f"idx: {i}, {descrp}")
    
    def to_df(self, attrs=[], extra_fn=[]): 
        """extra_fn other funcs that receive self and must return an numpy array of Dataset length"""
        header = ['index'] + listify(attrs) + [fn.__name__ for fn in listify(extra_fn)]
        attrs = [getattr(self, attr).numpy() for attr in listify(attrs)]
        extras = [fn(self) for fn in listify(extra_fn)]
        df =  pd.DataFrame([np.arange(len(intrp.ds)),*attrs, *extras])
        df = df.transpose()
        df.columns = header
        return df

class SegmentationTilePrecict_Old:
    """"old class that predicts only one image and requires to pass a learn object"""
    def __init__(self, img_path, learn, max_tile_size=512, max_tol=30, scale=1):
        self.img = open_image_cached(img_path, scale=scale)
        (self.rows, self.y_tile, self.y_diff), (self.cols, self.x_tile, self.x_diff) \
            = calc_n_tiles(self.img.size, max_tile_size, max_tol)
        self.learn = learn
        print(f"Image Prediction original size: ({self.img.size[0]}, {self.img.size[1]});"
            f"\nnumber of rows: {self.rows}, columns: {self.cols};"
            f"\nsize of tiles: ({self.y_tile}, {self.x_tile});"
            f"\ndiscared pixels due to rounding y: {self.y_diff} x: {self.x_diff}")

    def predict_mask(self, scale=1):
        # init mask, it is set to 0 because it is bigger than the sum of all tiles
        mask = torch.zeros(1, self.img.size[0], self.img.size[1], dtype=torch.int64)
        # Maybe need to use pred_batch for better performance
        for row, col in product(range(self.rows), range(self.cols)):
            img_tile = get_image_tile(self.img, (row, col), (self.y_tile, self.x_tile))
            mask_tile, _, _ = self.learn.predict(img_tile)
            mask[:, self.y_tile * row:self.y_tile * (row + 1),
            self.x_tile * col:self.x_tile * (col + 1)] = mask_tile.data
        return ImageSegment(mask)


class SegmentationTilePredictor:
    def __init__(self, learner_path, learner_file, tile_size, scale=3):
        self.learn = load_learner(learner_path,learner_file)
        self.scale = scale
        self.tile_size = tile_size
    def predict_mask(self, img_path):
        img = open_image_cached(img_path, scale=self.scale)
        rows = img.shape[1] // self.tile_size[0]
        cols = img.shape[2] // self.tile_size[1]

        # init mask, it is set to 0 because it is bigger than the sum of all tiles
        mask = torch.zeros(1, img.size[0], img.size[1], dtype=torch.int64)
        # Maybe need to use pred_batch for better performance
        for row, col in product(range(rows), range(cols)):
            img_tile = get_image_tile(img, (row, col), self.tile_size)
            mask_tile, _, _ = self.learn.predict(img_tile)
            mask[:, self.tile_size[0] * row : self.tile_size[0] * (row + 1),
            self.tile_size[1] * col : self.tile_size[1] * (col + 1)] = mask_tile.data
        return ImageSegment(mask), img


        
# %% tests

if __name__ == '__main__':
    segm = SemanticSegmentationTile(Path("dataset_segmentation/images"), Path("dataset_segmentation/labels"),
                                    max_tile_size=256, model=models.resnet18, scale=4, valid_split=.5, bs=2, max_tol=20,
                                    custom_bg_dir="backgrounds", num_bgs=1, bg_tfms=get_transforms()[0])


    # segm.learn = segm.learn.load("resnet18_4_epoch_freezed_1e-4_bg_0000")
    # print("predicting mask")
    # mask1 = segm.learn.predict(open_image("dataset_segmentation/images/albicocche1.png"))
    def plot_top_losses_pixel_diff(intrp, img_size, k=10, err_color=(1, 0, 0)):
        tloss = intrp.top_losses(img_size, k)
        fig, axes = plt.subplots(3, k // 3)
        for loss, i, ax in zip(tloss.values, tloss.indices, axes.flatten()):
            pred = intrp.pred_class[i]
            real = intrp.ds[i][1]
            img = intrp.ds[i][0]
            pred = image2np(pred[None])
            real = image2np(real.data)
            diff = pred != real
            img = image2np(img.data)
            img[diff == 1] = err_color
            ax.imshow(img)


    intrp = segm.interpet()
    intrp.plot_top_losses_pixel_diff(2)
    # mask, _, _ = segm.learn.predict(segm.data.valid_ds[0][0])
    # mask.show()
    # segm.learn.get_preds()
