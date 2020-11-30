import os
import cv2
import torch
import GENet
import pandas as pd
from torch import nn
import albumentations as albu

def main(opt):
    device = torch.device("cuda:0")

    current_checkpoint = [
            os.path.join("./artifacts_train", "checkpoints", opt.version, name)
            for name in os.listdir(os.path.join("./artifacts_train", "checkpoints", opt.version))
            if not name.startswith(".")
        ]
    checkpoint = sorted(current_checkpoint, key=lambda x: int(x.split("/")[-1].split("_")[0]), reverse=True)[0]
    print(f"Checkpoint: {checkpoint}")

    path_to_img = [os.path.join(opt.data_dir, name) for name in os.listdir(opt.data_dir) if not name.startswith(".")]

    val_trans = albu.Compose([
            albu.CenterCrop(512, 512),
            albu.Resize(256, 256),
            albu.Normalize() 
        ])

    model = GENet.genet_large(pretrained=False)
    model.fc_linear = nn.Linear(model.last_channels, 5, bias=True)
    model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    gpu = 0
    torch.cuda.set_device(gpu)
    model = model.cuda(0)
    model.eval()

    predicts = []
    for path in path_to_img:
        img = cv2.imread(path)
        img_inp = val_trans(image=img)["image"]
        img_inp = img_inp.transpose(2, 0, 1)
        img_inp = torch.from_numpy(img_inp)[None, ...]
        img_inp = img_inp.to(device)

        with torch.no_grad():
            logit = model(img_inp)

        _, predicted = torch.max(logit.data, 1)
        predicts += list(predicted.cpu().numpy())

    submission_df = pd.DataFrame(zip(path_to_img, predicts), columns=["image_id", "label"])
    submission_df["image_id"] = submission_df["image_id"].apply(lambda x: x.split("/")[-1])
    submission_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    from addict import Dict
    from config import submit_conf

    opt = Dict(submit_conf)
    print(opt)

    main(opt)