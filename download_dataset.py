from datasets import load_dataset

# Download ACNE04 from Hugging Face
dataset = load_dataset("Layered-Labs/ACNE04")

# Display dataset information
print(dataset)

# Access dataset
data = dataset["train"]

print("Total images:", len(data))

# Display one sample
sample = data[0]

print("Filename:", sample["file_name"])
print("Severity:", sample["severity"])
print("Severity level:", sample["level"])
print("Number of lesions:", sample["num_lesions_annotated"])

# Save the downloaded dataset locally
dataset.save_to_disk("data/ACNE04")

print("Dataset downloaded successfully!")
