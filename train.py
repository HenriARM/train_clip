"""
Source https://github.com/openai/CLIP/issues/83
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_csv=None, train=True, transform=None):
        df = pd.read_csv(data_csv)
        #df = df[:100]
        df_len = int(0.8 * len(df))
        if train is True:
            self.df = df[:df_len]
        else:
            self.df = df[df_len:]
        self.images1 = list(self.df["image_1"])
        self.images2 = list(self.df["image_2"])
        self.transform = transform

    def __getitem__(self, index):
        # Loading the image
        image1 = self.transform(Image.open(self.images1[index]))
        image2 = self.transform(Image.open(self.images2[index]))
        return image1, image2

    def __len__(self):
        return len(self.df)


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def main():
    data_csv = "data_3000.csv"
    # if using GPU then use mixed precision training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # must set jit=False for training
    # if device = cpu will return model.dtype = fp32, else fp16
    # see clip.model.convert_weights() https://github.com/openai/CLIP/blob/main/clip/model.py#L375
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print(preprocess)

    batch_size = 8  # must be larger than 1
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CustomDataset(data_csv, train=True, transform=preprocess)
    test_dataset = CustomDataset(data_csv, train=False, transform=preprocess)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, num_workers=4, batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, num_workers=4, batch_size=batch_size
    )

    loss = nn.CrossEntropyLoss()
    # params used from paper (page 48 https://arxiv.org/pdf/2103.00020.pdf)
    # try eps 1e-8
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    for epoch in range(epochs):
        model = model.train()

        for batch in train_dataloader:
            optimizer.zero_grad()
            image1, image2 = batch

            image1 = image1.to(device)
            image2 = image2.to(device)

            # forward
            image1_features = model.encode_image(image1)
            image2_features = model.encode_image(image2)

            # normalized features
            image1_features = image1_features / image1_features.norm(
                dim=1, keepdim=True
            )
            image2_features = image2_features / image2_features.norm(
                dim=1, keepdim=True
            )

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_image1 = logit_scale * image1_features @ image2_features.t()
            logits_image2 = logits_image1.t()

            ground_truth = torch.arange(len(image1), dtype=torch.long, device=device)
            total_loss = (loss(logits_image1, ground_truth) + loss(logits_image2, ground_truth)) / 2
            total_loss.backward()
            print(total_loss)
            if device == "cpu":
                optimizer.step()
            else:
                # convert model fp16 -> fp32 before updating weights
                convert_models_to_fp32(model)
                optimizer.step()
                # back fp32 -> fp16 https://github.com/openai/CLIP/blob/main/clip/model.py#L375
                clip.model.convert_weights(model)
            
            # taken from https://github.com/openai/CLIP/issues/83
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f"model_10.pt")


if __name__ == "__main__":
    main()


# TODO: use AdamW optimizer
# TODO: increase input resolution of backbones (CLIP ResNet version is modified and has ViT at the end)
# TODO:
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used


"""
Preprocess

Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
    CenterCrop(size=(224, 224))
    <function _convert_image_to_rgb at 0x7fc16ee38430>
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
)
"""
