from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


def dataloader(data_path, batch_size):

    data_transforms = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.ImageFolder(data_path, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return loader