from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights #######################################
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  #for mac gpu
print("Using device:", device)

BATCH_SIZE = 16   #how many photos in each step - train
IMG_SIZE = 224
NUM_WORKERS = 0    

train_dir = Path("train")
val_dir   = Path("val")
test_dir  = Path("test")

IMAGENET_MEAN = (0.485, 0.456, 0.406)  ##normalize format for resnet
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), #224x224
    transforms.RandomHorizontalFlip(p=0.5),   #flip sides
    transforms.RandomRotation(10),             #rotate
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  #colors
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfm = transforms.Compose([     #trans for val and test
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = ImageFolder(train_dir, transform=train_tfm)   #3 datasets
val_ds   = ImageFolder(val_dir,   transform=eval_tfm)
test_ds  = ImageFolder(test_dir,  transform=eval_tfm)

class_names = train_ds.classes   #taking classes name from dirs
num_classes = len(class_names)   #sum up how many classes there is
print("Classes:", class_names)   

#load to the model in groups
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -------- Model -------- #loading the resnet model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze all
for p in model.parameters():
    p.requires_grad = False

# Replace head
in_features = model.fc.in_features    #going to the last layer of resnet
model.fc = nn.Linear(in_features, num_classes)  #replacing the layer to our classes (before -there was a 1000 classes that resnet classified)
model = model.to(device)  #transfer the model to train on mac gpu

criterion = nn.CrossEntropyLoss()  #calculate loss func - How wrong the module was (higher - wrong)

def run_epoch(model, loader, optimizer=None):   #one epoch - all photos      ################func
    train = optimizer is not None               # optimizer- training
    model.train(train)                           #preaper to train

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:     #bringing photos
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()    # we are in a new train we want to clear the prev gradients

        logits = model(images)       #bringing the photos to resnet , getting scores
        loss = criterion(logits, labels)   #How wrong the module was -predict vs labels

        if train:     #if we are training and not val
            loss.backward() #calculate gradients
            optimizer.step() # doing the calculation - updating the weights

        total_loss += loss.item() * images.size(0)  #save the loss per batch (the loss num , the num of batch photos-16 , loss*16)
        preds = torch.argmax(logits, dim=1)         #taking the highst score for category (class)
        correct += (preds == labels).sum().item()   # checking predicts vs loss - numbers not math like loss
        total += images.size(0)

    return total_loss / total, correct / total     #avg wrong -loss , num of correct , total photos

best_val_acc = 0.0
best_state = None

start = time.time()

# -------- Stage 1: train only FC --------
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)    #training last layer - adam algoritem changing weights 
EPOCHS1 = 15    # train 15 times of all ds

print("\n--- Stage 1: train FC only ---")
for epoch in range(1, EPOCHS1 + 1):
    train_loss, train_acc = run_epoch(model, train_loader, optimizer=optimizer)   #finalyyyy train
    val_loss, val_acc     = run_epoch(model, val_loader,   optimizer=None)        #checking val without learning

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())  #taking the best weights from the best val

    print(f"Epoch {epoch:02d}/{EPOCHS1} | "
          f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
          f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
          f"best val {best_val_acc*100:.2f}%")

# -------- Stage 2: fine-tune layer4 + FC --------
print("\n--- Stage 2: fine-tune layer4 + FC ---")

for name, p in model.named_parameters():   #checking every parameter
    if name.startswith("layer4") or name.startswith("fc"):  # if it from layer 4 or fc
        p.requires_grad = True   #we can update it while training
    else:
        p.requires_grad = False

optimizer = torch.optim.Adam(                                     # new adam 
    filter(lambda p: p.requires_grad, model.parameters()),        # putting the parameters we can update
    lr=1e-4
)

EPOCHS2 = 8                    #stage 2 will run 8 epochs
for epoch in range(1, EPOCHS2 + 1):                                                   
    train_loss, train_acc = run_epoch(model, train_loader, optimizer=optimizer)    #layer4 training
    val_loss, val_acc     = run_epoch(model, val_loader,   optimizer=None)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

    print(f"[FT] Epoch {epoch:02d}/{EPOCHS2} | "
          f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
          f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
          f"best val {best_val_acc*100:.2f}%")

print("Training time (min):", (time.time() - start)/60)

# -------- Save best --------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
best_path = models_dir / "resnet18_best_ft.pt"

torch.save({
    "model_state_dict": best_state,
    "class_names": class_names,
    "img_size": IMG_SIZE,
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
}, best_path)

print("Saved:", best_path)

# -------- Test --------
model.load_state_dict(best_state)   #the best weights
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():     #because were not training 
    for images, labels in test_loader:    #checking all the test ds
        images = images.to(device)
        logits = model(images)             #model predicts scores
        preds = torch.argmax(logits, dim=1).cpu() #taking the highest - No need to convert logits to probabilities/percentages here because we are not calculating loss during testing.
        all_preds.extend(preds.tolist())      #preds list
        all_labels.extend(labels.tolist())     #correct list labels

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))
