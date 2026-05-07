from src_classifying.datasets.bloodmnist_dataset import BloodMNISTDataset
from torch.utils.data import DataLoader

from src_classifying.models.lightweight_cnn import LightweightCNN

path = './data/bloodmnist/bloodmnist.npz'

train_dataset = BloodMNISTDataset(path, split='train')
val_dataset = BloodMNISTDataset(path, split='val')
test_dataset = BloodMNISTDataset(path, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = LightweightCNN()

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    print(labels[:10])
    print("=========================")
    outputs = model(images)
    print(outputs.shape)
    break

print()


