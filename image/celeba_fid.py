from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from fid import FID


class CelebAFID(FID):
    def __init__(self, batch_size=256, data_name='celeba',
                 workers=0, verbose=True):
        self.batch_size = batch_size
        self.workers = workers
        super().__init__(data_name, verbose)

    def complete_data(self):
        data = datasets.ImageFolder(
            'celeba',
            transforms.Compose([
                transforms.CenterCrop(108),
                transforms.Resize(size=64, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
            ]))

        images = len(data)
        data_loader = DataLoader(
            data, batch_size=self.batch_size, num_workers=self.workers)

        return data_loader, images
