import cv2
import numpy as np
import torch
from .road_model.model import FusePlusSatDeepLabV3Plus
from torchvision import transforms
from ultralytics import YOLO


def img_preproc_road(img):
    img = cv2.resize(img, (1024, 1024))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    img = transform(img)

    return img[None, ...]


def load_ckpt_eval(ckpt_path, map_location="cuda:0"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    ckpt = ckpt['sat_state_dict']

    model = FusePlusSatDeepLabV3Plus(in_channels=3,
                                     encoder_name="resnet34",
                                     encoder_weights="imagenet",
                                     classes=1,
                                     activation="sigmoid",
                                     use_dsam=False)
    model.load_state_dict(ckpt)
    model = model.to(map_location)
    model.eval()
    return model


def inference_image_road(img_path, ckpt_path, map_location="cuda:0"):
    input = img_preproc_road(img_path).to(map_location)
    model = load_ckpt_eval(ckpt_path)

    with torch.no_grad():
        output, _, _ = model(input)
    output = output.squeeze(1).cpu().numpy()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    output = output.astype(np.int32)

    return output.squeeze()


def inference_image_vegetation(img_path, ckpt_path, map_location="cuda:0"):
    model = YOLO(ckpt_path)

    results = model(img_path, device=map_location)
    veg_contour = []
    for result in results:
        if result.masks:
            veg_contour+=result.masks.xy
    return veg_contour

# img_path='/fast/zcb/code/mdx/gen3d_realCity/src_test.jpg'
# ckpt_path='/fast/zcb/code/mdx/gen3d_realCity/bg_extract/ckpt/p2cnet_road.pth'
# res=inference_image_road(img_path, ckpt_path)