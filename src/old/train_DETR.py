# Imports
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
import albumentations
import numpy as np
from huggingface_hub import login
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Constants
ACCESS_TOKEN_WRITE = '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'
DATASET_DIR = r"C:\Users\chris\foosball-dataset\merged_foosball_dataset"
# DATASET_DIR = r"C:\Users\chris\foosball-dataset\foosball_coco_3"
CHECKPOINT = "facebook/detr-resnet-50"
CATEGORIES = ['ball', 'figure']
OUTPUT_DIR = "detr-resnet-50_finetuned_merged_foosball_dataset"


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": labels}
    return batch


# Initial Setup
login(token=ACCESS_TOKEN_WRITE)
foosball_dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)
id2label = {index: x for index, x in enumerate(CATEGORIES, start=0)}
label2id = {v: k for k, v in id2label.items()}

# Image Processor
image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# Transformations
transform = albumentations.Compose(
    [
        albumentations.Resize(720, 1280)
#         albumentations.HorizontalFlip(p=0.5),
#         albumentations.RandomBrightnessContrast(p=0.75),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

# Apply Transformations to the Dataset
foosball_dataset["train"] = foosball_dataset["train"].with_transform(transform_aug_ann)

# Model
model = AutoModelForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=foosball_dataset["train"],
    tokenizer=image_processor,
)

# Training
trainer.train()
