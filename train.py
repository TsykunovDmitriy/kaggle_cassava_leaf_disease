import os
import cv2
import timm
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from losses import get_loss
import albumentations as albu
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from torch.utils.data.sampler import WeightedRandomSampler

rng = np.random.default_rng()

def pgd_attack(
            x, 
            label, 
            model, 
            criterion,
            alpha=0.05, 
            attack_eps=0.1, 
            attack_steps=2, 
            random_init=False, 
            target=None, 
            clamp=(0, 1),
        ):
    
    x_adv = x.clone()

    if random_init:
        # Flag to use random initialization
        x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps

    for i in range(attack_steps):
        x_adv.requires_grad = True

        model.zero_grad()
        logits = model(x_adv)

        if target is None:
            # Untargeted attacks - gradient ascent
            loss = criterion(logits, label)
            loss.backward()
            grad = x_adv.grad.detach()
            grad = grad.sign()
            x_adv = x_adv + alpha * grad

        else:
            # Targeted attacks - gradient descent
            assert target.size() == label.size()
            loss = criterion(logits, label)
            loss.backward()
            grad = x_adv.grad.detach()
            grad = grad.sign()
            x_adv = x_adv - alpha * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = x_adv.detach()

    return x_adv

class LeafData(Dataset):
    def __init__(self, df, transforms=None, tta=None):
        self.tta = tta
        self.data = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data.iloc[idx]
        img = cv2.imread(path)
        
        if self.tta:
            images = []
            for _ in range(self.tta):
                if self.transforms:
                    images.append(self.transforms(image=img)["image"])

            images = [img.transpose(2, 0, 1) for img in images]
            images = [torch.from_numpy(img) for img in images]

            return images, label

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return torch.from_numpy(img.transpose(2, 0, 1)), label

def get_weights(df):
    value_counts = df["label"].value_counts()
    return [1/value_counts[df["label"][i]] for i in range(len(df))]

def main(opt):
    torch.manual_seed(opt.seed)

    if os.path.isdir("./artifacts_train"):
        log_versions = [
            int(name.split("_")[-1]) 
            for name in os.listdir(os.path.join("./artifacts_train", "logs")) 
            if os.path.isdir(os.path.join("./artifacts_train", "logs", name))
        ]
        current_version = f"version_{max(log_versions) + 1}"
    else:
        os.makedirs(os.path.join("./artifacts_train", "logs"), exist_ok=True)
        os.makedirs(os.path.join("./artifacts_train", "checkpoints"), exist_ok=True)
        current_version = "version_0"
    logger = SummaryWriter(logdir=os.path.join("./artifacts_train", "logs", current_version))
    os.makedirs(os.path.join("./artifacts_train", "checkpoints", current_version), exist_ok=True)

    device = torch.device("cuda:0")

    # Train Val Split
    path_to_train = os.path.join(opt.data_dir, "train_images/")
    train_df = pd.read_csv(os.path.join(opt.data_dir, "train.csv"))
    train_df["image_id"] = train_df["image_id"].apply(lambda x: os.path.join(path_to_train, x))
    train_df = train_df.sample(frac=1, random_state=opt.seed).reset_index(drop=True)
    train_df.columns = ["path", "label"]

    val_df = train_df.loc[int(len(train_df)*opt.train_split/100):].reset_index(drop=True)
    train_df = train_df.loc[:int(len(train_df)*opt.train_split/100)].reset_index(drop=True)

    # Augmentations
    train_trans = albu.Compose([
            albu.RandomResizedCrop(*opt.input_shape),
            albu.VerticalFlip(),
            albu.HorizontalFlip(),
            albu.Transpose(p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albu.CoarseDropout(p=0.5),
            albu.Cutout(p=0.5),
            albu.Normalize()
        ])

    val_trans = albu.Compose([
            albu.RandomResizedCrop(*opt.input_shape),
            albu.VerticalFlip(),
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albu.Normalize() 
        ])

    # Dataset init
    data_train = LeafData(train_df, transforms=train_trans)
    data_val = LeafData(val_df, transforms=val_trans, tta=opt.tta)
    weights = get_weights(train_df)
    sampler_train = WeightedRandomSampler(weights, len(data_train))
    dataloader_train = DataLoader(data_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    dataloader_val = DataLoader(data_val, shuffle=True, batch_size=8, num_workers=opt.num_workers)

    # Model init
    model = timm.create_model(opt.model_arch, pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model.to(device)

    # freeze first opt.freeze_percent params
    param_count = len(list(model.parameters()))
    for param in list(model.parameters())[:int(param_count*opt.freeze_percent/100)]:
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=opt.num_epoch, T_mult=1, eta_min=1e-6, last_epoch=-1)
    criterion = get_loss(opt)

    best_acc = 0
    iteration_per_epoch = opt.iteration_per_epoch if opt.iteration_per_epoch else len(dataloader_train)
    for epoch in range(opt.num_epoch):
        # Train
        model.train()
        dataloader_iterator = iter(dataloader_train)
        pbar = tqdm(range(iteration_per_epoch), desc=f"Train : Epoch: {epoch + 1}/{opt.num_epoch}")
        
        for step in pbar:     
            try:
                images, labels = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(dataloader_train)
                images, labels = next(dataloader_iterator)
            
            images = images.to(device)
            labels = labels.to(device)

            if opt.adversarial_attack:
                idx_for_attack = list(rng.choice(labels.size(0), size=labels.size(0) // 4))
                images[idx_for_attack] = pgd_attack(images[idx_for_attack], labels[idx_for_attack], model, criterion)
            
            logit = model(images)
            loss = criterion(logit, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + step/iteration_per_epoch)
            
            _, predicted = torch.max(logit.data, 1)
            accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
            pbar.set_postfix({"Accuracy": accuracy, "Loss": loss.cpu().data.numpy().item(), "LR": optimizer.param_groups[0]["lr"]})
            logger.add_scalar('Loss/Train', loss.cpu().data.numpy().item(), epoch*iteration_per_epoch + step + 1)
            logger.add_scalar('Accuracy/Train', accuracy, epoch*iteration_per_epoch + step + 1)
            logger.add_scalar('LR/Train', optimizer.param_groups[0]["lr"], epoch*iteration_per_epoch + step + 1)
        
        # Val
        print(f"Eval start! Epoch {epoch + 1}/{opt.num_epoch}")
        correct = 0
        total = 0
        loss_sum = 0

        model.eval()
        dataloader_iterator = iter(dataloader_val)
        pbar = tqdm(range(len(dataloader_val)), desc=f"Eval : Epoch: {epoch + 1}/{opt.num_epoch}")
        for step in pbar: 
            try:
                images, labels = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(dataloader_val)
                images, labels = next(dataloader_iterator)

            labels = labels.to(device)

            if opt.tta:
                predicts = []
                loss_tta = 0
                for i in range(opt.tta):
                    img = images[i].to(device)
            
                    with torch.no_grad():
                        logit = model(img)

                    loss = criterion(logit, labels)
                    loss_tta += loss.cpu().data.numpy().item() / opt.tta
                    predicts.append(F.softmax(logit, dim=-1)[None, ...])
                
                predicts = torch.cat(predicts, dim=0).mean(dim=0)
                loss_sum += loss_tta

            else:
                images = images.to(device)
                with torch.no_grad():
                    logit = model(images)

                loss = criterion(logit, labels)
                predicts = F.softmax(logit, dim=-1)
                loss_sum += loss.cpu().data.numpy().item()

            #accuracy
            _, predicted = torch.max(predicts.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"Accuracy": 100 * (predicted == labels).sum().item() / labels.size(0), "Loss": loss.cpu().data.numpy().item()})
            
        accuracy = 100 * correct / total
        loss_mean = loss_sum / len(dataloader_val)
        logger.add_scalar('Loss/Val', loss_mean, epoch*iteration_per_epoch + step + 1)
        logger.add_scalar('Accuracy/Val', accuracy, epoch*iteration_per_epoch + step + 1)
        print(f"Epoch: {epoch + 1}, Accuracy: {accuracy:.5f}, Loss {loss_mean:.5f}")
        if accuracy > best_acc:
            print("Saved checkpoint!")
            best_acc = accuracy
            torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "accuracy": round(accuracy, 5),
                    "loss": round(loss_mean, 5),
                    "config": opt,
                }, 
                os.path.join("./artifacts_train", "checkpoints", current_version, f"{epoch + 1}_accuracy_{accuracy:.5f}.pth"))

if __name__ == "__main__":
    from addict import Dict
    from config import train_config

    opt = Dict(train_config)
    print(opt)

    main(opt)