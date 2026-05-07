from torchvision import transforms
from torch.utils.data import DataLoader

from src_counting.datasets.bbbc005_dataset import BBBC005Dataset


IMAGE_DIR = "./data/bbbc005/BBBC005_v1_images"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = BBBC005Dataset(
    image_dir=IMAGE_DIR,
    transform=transform,
    stain=2,
    blur_levels=[1],
)

print("Dataset size:", len(dataset))

image, count = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min().item())
print("Image max:", image.max().item())
print("Count:", count.item())

loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, counts = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch count shape:", counts.shape)
print("Batch counts:", counts.squeeze().tolist())