from  data.loader import train_dataloader


if __name__ == "__main__":
    for imgs, labels in train_dataloader:
        print(imgs, labels)
        print(imgs.shape)
        break