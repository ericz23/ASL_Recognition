import os

dataset_path = "dataset/asl_alphabet_train/asl_alphabet_train"
letters = sorted(os.listdir(dataset_path))

print("Dataset contains the following categories:")
print(letters)
