from datasets import Dataset, concatenate_datasets
from pathlib import Path

shard_dir = Path("data/ACNE04/train")
output_dir = shard_dir / "images"   # data/ACNE04/train/images/
output_dir.mkdir(parents=True, exist_ok=True)

# Load and combine all shard .arrow files (sorted for stable order)
shards = sorted(shard_dir.glob("data-*-of-*.arrow"))
dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in shards])

print("Dataset features:", dataset.features)
print("Shards loaded:", len(shards))
print("Total records:", len(dataset))

# Extract images
for index, sample in enumerate(dataset):
    image = sample["image"].convert("RGB")
    image.save(output_dir / f"acne_{index:05d}.jpg", quality=95)

print(f"Successfully extracted {len(dataset)} images!")
