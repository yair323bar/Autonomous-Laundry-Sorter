from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

train_dir = Path("train")

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

ds = ImageFolder(train_dir, transform=tfm)

print("Num images:", len(ds))
print("Classes:", ds.classes)
print("Class->index:", ds.class_to_idx)

x, y = ds[0]
print("One sample shape:", x.shape, "label:", y)
