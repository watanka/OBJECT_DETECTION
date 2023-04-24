import albumentations as A
import albumentations.pytorch as pytorch

class DataAug :
    def __init__(self, cfg) :
        self.train_transform = A.Compose(
            [   
                A.Normalize(mean = cfg.aug.mean, std = cfg.aug.std, max_pixel_value = 255), # dataloader read image with 0-255, and we want the value to be range from 0 to 1.
                A.geometric.resize.RandomScale(scale_limit=cfg.aug.scale_limit),
                A.geometric.transforms.Affine(translate_percent = [cfg.aug.translation, cfg.aug.translation]), 
                A.geometric.resize.SmallestMaxSize(max_size=int(cfg.aug.scale * cfg.model.img_size)),
                A.transforms.ColorJitter(brightness = cfg.aug.brightness, contrast = 0.6, saturation = cfg.aug.saturation, hue = 0.6, p = 0.6), 
                A.RandomCrop(width=cfg.model.img_size, height=cfg.model.img_size, always_apply=True, p=1.0),
                A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None), 
                pytorch.transforms.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["label"], min_visibility=cfg.aug.min_visibility # bounding box will be changed into yolo format after the encoding
            ),
        )

        self.test_transform = A.Compose(
                [
                    A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
                    A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
                    A.Normalize(mean = cfg.aug.mean, std = cfg.aug.std, max_pixel_value = 255), # dataloader read image with 0-255, and we want the value to be range from 0 to 1.
                    pytorch.transforms.ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["label"], min_visibility=0.5
                ),
            )

        self.predict_transform = A.Compose(
                    [
                    A.geometric.resize.LongestMaxSize(max_size=cfg.model.img_size),
                    A.PadIfNeeded(min_width=cfg.model.img_size, min_height=cfg.model.img_size, border_mode=None),
                    A.Normalize(mean = cfg.aug.mean, std = cfg.aug.std, max_pixel_value = 255), # dataloader read image with 0-255, and we want the value to be range from 0 to 1.
                    pytorch.transforms.ToTensorV2(),
                ],
            )