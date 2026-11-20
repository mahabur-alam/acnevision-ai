"""Download the ACNE04 dataset from Hugging Face into data/raw/.

ACNE04 is released for ACADEMIC / RESEARCH USE ONLY (non-commercial).
See docs/dataset_documentation.md before using the data.

After running this, the Arrow shards land under data/raw/ACNE04/train/, which is
where scripts/convert_acne04.py looks by default.
"""

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

# Save the downloaded dataset locally. save_to_disk writes the shards to
# data/raw/ACNE04/train/data-*-of-*.arrow, matching the conversion default.
dataset.save_to_disk("data/raw/ACNE04")

print("Dataset downloaded successfully!")
print("Next: python scripts/convert_acne04.py")
