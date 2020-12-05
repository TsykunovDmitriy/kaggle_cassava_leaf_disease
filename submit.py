import os
import cv2
import timm
import torch
import pandas as pd
from torch import nn
import albumentations as albu
import torch.nn.functional as F

def main(opt):
    device = torch.device("cuda")

    current_checkpoint = [
            os.path.join("./artifacts_train", "checkpoints", opt.version, name)
            for name in os.listdir(os.path.join("./artifacts_train", "checkpoints", opt.version))
            if not name.startswith(".")
        ]
    checkpoint = sorted(current_checkpoint, key=lambda x: int(x.split("/")[-1].split("_")[0]), reverse=True)[0]
    print(f"Checkpoint: {checkpoint}")

    path_to_img = [os.path.join(opt.data_dir, name) for name in os.listdir(opt.data_dir) if not name.startswith(".")]

    val_trans = albu.Compose([
            albu.RandomResizedCrop(*opt.input_shape),
            albu.VerticalFlip(),
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albu.Normalize()
        ])

    model = timm.create_model(opt.model_arch, pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    predicts = []
    for path in path_to_img:
        img = cv2.imread(path)
        tta_preds = []
        for _ in range(opt.tta):
            img_inp = val_trans(image=img)["image"]
            img_inp = img_inp.transpose(2, 0, 1)
            img_inp = torch.from_numpy(img_inp)[None, ...]
            img_inp = img_inp.to(device)

            with torch.no_grad():
                logit = model(img_inp)
            
            tta_preds.append(F.softmax(logit, dim=-1)[None, ...])
        
        tta_pred = torch.cat(tta_preds, dim=0).mean(dim=0)

        _, predicted = torch.max(tta_pred.data, 1)
        predicts += list(predicted.cpu().numpy())

    submission_df = pd.DataFrame(zip(path_to_img, predicts), columns=["image_id", "label"])
    submission_df["image_id"] = submission_df["image_id"].apply(lambda x: x.split("/")[-1])
    submission_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    from addict import Dict
    from config import submit_config

    opt = Dict(submit_config)
    print(opt)

    main(opt)