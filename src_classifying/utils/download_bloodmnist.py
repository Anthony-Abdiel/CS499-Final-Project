import medmnist
from medmnist import INFO

data_flag = 'bloodmnist'
info = INFO[data_flag]

DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=True, root='./data/bloodmnist')
val_dataset = DataClass(split='val', download=True, root='./data/bloodmnist')
test_dataset = DataClass(split='test', download=True, root='./data/bloodmnist')

print("Download complete!")
print("Train: ", len(train_dataset))
print("Val: ", len(val_dataset))
print("Test: ", len(test_dataset))
