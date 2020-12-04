from torch.utils.data import Dataset


class EventDataset(Dataset):
    def __init__(self, image, image_label, transform=None):
        self.image = image
        self.image_label = image_label
        self.transform = transform

    def __getitem__(self, idx):
        trans_image = self.image[idx]
        trans_image_label = self.image_label[idx]
        if self.transform:
            trans_image = self.transform(trans_image)
            trans_image_label = self.transform(trans_image_label)
        return trans_image, trans_image_label

    def __len__(self):
        return len(self.image)


if __name__ == '__main__':
    pass
