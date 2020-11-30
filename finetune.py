import os
import cv2
import torch
import GENet
import pandas as pd
from torch import nn
from tqdm import tqdm
# import pretrainedmodels
from losses import get_loss
import albumentations as albu
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data.sampler import WeightedRandomSampler

class LiefData(Dataset):
    def __init__(self, df, transforms=None):
        self.data = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data.iloc[idx]
        img = cv2.imread(path)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img, label

def get_weights(df):
    value_counts = df["label"].value_counts()
    return [1/value_counts[df["label"][i]] for i in range(len(df))]

def main(opt):
    torch.manual_seed(opt.seed)
    assert os.path.isdir("./artifacts_train"), "No train artifacts"
    assert os.path.isdir(os.path.join("./artifacts_train", "checkpoints", opt.finetune_version)), f"No found checkpoint {opt.finetune_version}"

    os.makedirs(os.path.join("./artifacts_finetune", "logs"), exist_ok=True)
    os.makedirs(os.path.join("./artifacts_finetune", "checkpoints"), exist_ok=True)
    
    logger = SummaryWriter(logdir=os.path.join("./artifacts_finetune", "logs", opt.finetune_version))
    current_checkpoint = [
            os.path.join("./artifacts_train", "checkpoints", opt.finetune_version, name)
            for name in os.listdir(os.path.join("./artifacts_train", "checkpoints", opt.finetune_version))
            if not name.startswith(".")
        ]
    current_checkpoint = sorted(current_checkpoint, key=lambda x: int(x.split("/")[-1].split("_")[0]), reverse=True)[0]
    print(f"Finetuning checkpoint: {current_checkpoint}")

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
            albu.CenterCrop(512, 512),
            albu.Resize(256, 256),
            albu.Normalize() 
        ])

    val_trans = albu.Compose([
            albu.CenterCrop(512, 512),
            albu.Resize(256, 256),
            albu.Normalize() 
        ])

    # Dataset init
    data_train = LiefData(train_df, transforms=train_trans)
    data_val = LiefData(val_df, transforms=val_trans)
    weights = get_weights(train_df)
    sampler_train = WeightedRandomSampler(weights, len(data_train))
    dataloader_train = DataLoader(data_train, batch_size=opt.batch_size, sampler=sampler_train, num_workers=opt.num_workers)
    dataloader_val = DataLoader(data_val, shuffle=True, batch_size=8, num_workers=opt.num_workers)

    # Model init
    model = GENet.genet_large(pretrained=True, root=opt.genet_checkpoint)
    model.fc_linear = nn.Linear(model.last_channels, 5, bias=True)
    model.load_state_dict(torch.load(current_checkpoint)["model_state_dict"])
    gpu = 0
    torch.cuda.set_device(gpu)
    model = model.cuda(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=4e-5)
    criterion = get_loss(opt)

    best_acc = 0
    iteration_per_epoch = iteration_per_epoch if opt.iteration_per_epoch else len(dataloader_train)
    for epoch in range(5):
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
            
            logit = model(images)
            loss = criterion(logit, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logit.data, 1)
            accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
            pbar.set_postfix({"Accuracy": accuracy, "Loss": loss.cpu().data.numpy().item()})
            logger.add_scalar('Loss/Train', loss.cpu().data.numpy().item(), epoch*iteration_per_epoch + step + 1)
            logger.add_scalar('Accuracy/Train', accuracy, epoch*iteration_per_epoch + step + 1)
        
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

            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                logit = model(images)

            loss = criterion(logit, labels)
            loss_sum += loss.cpu().data.numpy().item()

            #accuracy
            _, predicted = torch.max(logit.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"Accuracy": (predicted == labels).sum().item() / labels.size(0), "Loss": loss.cpu().data.numpy().item()})
            
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
                    # "scheduler": scheduler.state_dict(),
                    "accuracy": round(accuracy, 5),
                    "loss": round(loss_mean, 5),
                    "config": opt,
                }, 
                os.path.join("./artifacts_train", "checkpoints", opt.finetune_version, f"{epoch + 1}_accuracy_{accuracy:.5f}.pth"))
        # scheduler.step()

if __name__ == "__main__":
    from addict import Dict
    from config import train_config

    opt = Dict(train_config)
    print(opt)

    main(opt)
