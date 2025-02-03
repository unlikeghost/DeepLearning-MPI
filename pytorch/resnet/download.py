import torchvision
import torchvision.transforms as transforms

# Define root directory for the dataset
dataset_root = "./data"

# Use a simple transformation (normalize + ToTensor)
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# # Download training and test datasets
print("Downloading CIFAR-10 dataset...")
train_set = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
print("Dataset downloaded successfully!")
