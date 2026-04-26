from torchvision import datasets, transforms

DATA_DIR = "../data/01_raw/COVID-19_Radiography_Dataset"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Número de imágenes:", len(dataset))
print("Clases:", dataset.classes)