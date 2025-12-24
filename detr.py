import os
import wandb
from datetime import datetime
import torch
import numpy as np
import supervision as sv
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataclasses import dataclass, replace

# Define constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"  # RT-DETR pre-trained checkpoint

# Create PyTorch dataset for RT-DETR
class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]

        # Convert image to RGB numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure proper RGB order
        if image.shape[2] == 3:  # Only convert if it's an RGB image
            image = image[:, :, ::-1]  # BGR to RGB
            
        boxes = annotations.xyxy
        categories = annotations.class_id

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category=categories
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes)
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt")

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result

# Collate function for DataLoader
def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

# MAP Evaluator for metrics calculation
@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, (height, width) in zip(target_batch, image_size_batch):
                boxes = target["boxes"]
                boxes = sv.xcycwh_to_xyxy(boxes)
                boxes = boxes * np.array([width, height, width, height])
                boxes = torch.tensor(boxes)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):
        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
        return metrics

def main():
    # Define direct paths and parameters
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    run_name = args.wandb_name if args.wandb_name else f"rt-detr-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Define simplified training parameters
    train_params = {
        'dataset_dir': args.dataset_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'lr': args.lr,
        'output_dir': args.output_dir,
        'early_stopping': args.early_stopping,
        'wandb': True,  # Enable wandb logging
        'project': args.wandb_project,
        'run': run_name
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(train_params['output_dir'], exist_ok=True)
    
    print(f"Training RT-DETR model on dataset: {train_params['dataset_dir']}")
    print(f"Training for {train_params['epochs']} epochs with batch size {train_params['batch_size']}")
    
    # Load dataset
    print("Loading dataset...")
    
    # Load the dataset using supervision's DetectionDataset
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=f"{train_params['dataset_dir']}/train",
        annotations_path=f"{train_params['dataset_dir']}/train/_annotations.coco.json",
    )
    
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=f"{train_params['dataset_dir']}/valid",
        annotations_path=f"{train_params['dataset_dir']}/valid/_annotations.coco.json",
    )
    
    print(f"Number of training images: {len(ds_train)}")
    print(f"Number of validation images: {len(ds_valid)}")
    
    # Setup image processor and model
    print("Setting up RT-DETR model...")
    
    # Define image size
    image_size = 640
    
    # Initialize image processor
    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        do_resize=True,
        size={"width": image_size, "height": image_size},
    )
    
    # Set up data augmentation
    train_augmentation_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=25
        ),
    )

    valid_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=1
        ),
    )
    
    # Prepare datasets
    pytorch_dataset_train = PyTorchDetectionDataset(
        ds_train, processor, transform=train_augmentation_and_transform)
    pytorch_dataset_valid = PyTorchDetectionDataset(
        ds_valid, processor, transform=valid_transform)
    
    # Setup id2label and label2id mapping
    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}
    
    print(f"Classes: {id2label}")
    
    # Load RT-DETR model
    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    )
    
    # Setup evaluation function
    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=processor, 
        threshold=0.01, 
        id2label=id2label
    )
    
    # Initialize wandb if specified
    if train_params['wandb']:
        wandb.init(
            project=train_params['project'],
            name=train_params['run'],
            config={
                "model": "RT-DETR",
                "epochs": train_params['epochs'],
                "batch_size": train_params['batch_size'],
                "grad_accum_steps": train_params['grad_accum_steps'],
                "image_size": image_size,
                "learning_rate": train_params['lr'],
                "dataset": train_params['dataset_dir'],
            }
        )
    
    # Configure simplified training arguments
    training_args = TrainingArguments(
        output_dir=train_params['output_dir'],
        num_train_epochs=train_params['epochs'],
        max_grad_norm=0.1,
        learning_rate=train_params['lr'],
        warmup_steps=300,
        per_device_train_batch_size=train_params['batch_size'],
        gradient_accumulation_steps=train_params['grad_accum_steps'],
        dataloader_num_workers=2,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=train_params['early_stopping'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="wandb" if train_params['wandb'] else "none",
        run_name=train_params['run'] if train_params['wandb'] else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    # Train the model
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save the final model and processor
    model.save_pretrained(f"{train_params['output_dir']}/final")
    processor.save_pretrained(f"{train_params['output_dir']}/final")
    
    print(f"Training completed. Model saved to {train_params['output_dir']}")
    
    # Close wandb if it was used
    if train_params['wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()
