from pathlib import Path
from fastai.vision import *

path = Path("dataset_segmentation/")
path_lbl = path / "labels"
path_img = path / "images"
test_img = path_img / "albicocche1.png"

inf_learn = load_learner(path)


def get_data_empty(size):
    return (SegmentationItemList.from_folder("")
            .split_none()
            .label_empty()
            .transform(size=size)
            .databunch(bs=1)
            .normalize(imagenet_stats))


def pred_split_img(img_path, learner, max_size=(512, 512)):
    img = open_image(img_path)
    x_max, y_max = max_size
    _, x_sz, y_sz = img.shape
    pred_mask = torch.empty((1, x_sz, y_sz), dtype=torch.int64)

    # learner.data = get_data(max_size, path_img/"1_img", 1) #uglyy hack, TODO find a way to create empty databunch

    for i_y in range(y_sz // y_max + 1):
        y_start = i_y * y_max
        y_end = y_start + y_max
        curr_y_sz = y_max
        if y_end > y_sz:  # manage last block
            y_end = y_sz
            curr_y_sz = y_sz % y_max

        for i_x in range(x_sz // x_max + 1):
            x_start = i_x * x_max
            x_end = x_start + x_max
            curr_x_sz = x_max
            if x_end > x_sz:  # manage last block
                x_end = x_sz
                curr_x_sz = x_sz % x_max

            learner.data = get_data_empty((curr_x_sz, curr_y_sz))
            crop_img = Image(img.data[:, x_start:x_end, y_start:y_end])
            crop_mask, _, _ = learner.predict(crop_img)
            pred_mask[:, x_start:x_end, y_start:y_end] = crop_mask.data

    ImageSegment(pred_mask).show()
#     pred_mask, _, _ = inf_learn.predict(img)
#     real_mask = open_mask(real_mask_path, div=True)
#     real_mask = image2np(real_mask.data)
#     pred_mask = image2np(pred_mask.data)
