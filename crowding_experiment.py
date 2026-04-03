"""
Combined train + eval for crowding experiment.
Trains decoders on outside images, then immediately evaluates on inside images
without saving/loading the model (avoids all save/load bugs).
Run: python crowding_experiment.py
"""

import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from src.utils.dataset_utils import add_compute_stats, ImageDatasetAnnotations, AffineTransform
from src.utils.device_utils import set_global_device, to_global_device
from src.utils.net_utils import GrabNet
import torchvision

# ============================================================
# Config
# ============================================================
ANNOTATION = "data/low_mid_level_vision/un_crowding/annotation.csv"
OUTPUT_DIR = Path("results/decoder/crowding/eval")
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5

# ============================================================
# Setup
# ============================================================
if __name__ == '__main__':
    set_global_device(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Create datasets manually (bypass get_dataloader issues)
    # ============================================================
    def make_loader(filter_value, name, return_path=False):
        ds = ImageDatasetAnnotations(
            task_type="classification",
            csv_file=ANNOTATION,
            img_path_col="Path",
            label_cols="VernierType",
            filters={"VernierInOut": filter_value},
            return_path=return_path,
        )
        ds.name = name
        
        # ImageNet normalization
        ds.stats = {"mean": [0.491, 0.482, 0.44], "std": [0.247, 0.243, 0.262]}
        ds.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=ds.stats["mean"], std=ds.stats["std"]),
        ])
        
        loader = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=(not return_path),
            num_workers=0, pin_memory=True,
        )
        return loader

    train_loader = make_loader("outside", "train_outside")
    test_inside_loader = make_loader("inside", "test_inside", return_path=True)
    test_outside_loader = make_loader("outside", "test_outside", return_path=True)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test inside samples: {len(test_inside_loader.dataset)}")
    print(f"Test outside samples: {len(test_outside_loader.dataset)}")

    # ============================================================
    # Build model
    # ============================================================
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes} -> {train_loader.dataset.classes}")

    net, _, _ = GrabNet.get_net(
        "resnet152_decoder",
        imagenet_pt=True,
        num_classes=num_classes,
    )

    # Freeze backbone, only train decoders
    for param in net.parameters():
        param.requires_grad = False
    for param in net.decoders.parameters():
        param.requires_grad = True

    net = to_global_device(net)
    net.train()

    num_decoders = len(net.decoders)
    print(f"Number of decoders: {num_decoders}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizers = [
        torch.optim.Adam(net.decoders[i].parameters(), lr=LEARNING_RATE)
        for i in range(num_decoders)
    ]

    # ============================================================
    # Train
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        total_correct = [0] * num_decoders
        total_samples = 0

        for images, labels in train_loader:
            images = to_global_device(images)
            labels = to_global_device(labels)

            [opt.zero_grad() for opt in optimizers]
            out_dec = net(images)

            loss = to_global_device(torch.tensor(0.0, requires_grad=True))
            for d in range(num_decoders):
                loss = loss + loss_fn(out_dec[d], labels)

            loss.backward()
            [opt.step() for opt in optimizers]

            # Track accuracy
            for d in range(num_decoders):
                preds = torch.argmax(out_dec[d], dim=1)
                total_correct[d] += (preds == labels).sum().item()
            total_samples += len(labels)

        accs = [total_correct[d] / total_samples for d in range(num_decoders)]
        acc_str = " / ".join([f"dec{d}:{accs[d]:.3f}" for d in range(num_decoders)])
        print(f"Epoch {epoch:2d}: loss={loss.item():.4f}  {acc_str}")

    # ============================================================
    # Evaluate (immediately, no save/load)
    # ============================================================
    def evaluate_loader(loader, name):
        results = []
        total_correct = [0] * num_decoders
        total_samples = 0

        with torch.no_grad():
            for data in tqdm(loader, desc=name):
                if len(data) == 3:
                    images, labels, paths = data
                else:
                    images, labels = data
                    paths = [""] * len(labels)

                images = to_global_device(images)
                labels_dev = to_global_device(labels)
                out_dec = net(images)

                for i in range(len(labels)):
                    row = {"image_path": paths[i] if isinstance(paths[i], str) else "", 
                           "label": labels[i].item()}
                    for d in range(num_decoders):
                        pred = torch.argmax(out_dec[d][i]).item()
                        row[f"prediction_dec_{d}"] = pred
                    results.append(row)

                for d in range(num_decoders):
                    preds = torch.argmax(out_dec[d], dim=1)
                    total_correct[d] += (preds == labels_dev).sum().item()
                total_samples += len(labels)

        df = pd.DataFrame(results)
        
        # Save
        save_dir = OUTPUT_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / "predictions.csv", index=False)

        # Print
        print(f"\n=== {name} ===")
        for d in range(num_decoders):
            acc = total_correct[d] / total_samples
            pred_dist = df[f"prediction_dec_{d}"].value_counts().to_dict()
            print(f"  Decoder {d}: acc={acc:.3f}  predictions={pred_dist}")

        return df

    print("\n" + "=" * 60)
    print("EVALUATING OUTSIDE (training data)")
    print("=" * 60)
    df_outside = evaluate_loader(test_outside_loader, "outside_vernier_test")

    print("\n" + "=" * 60)
    print("EVALUATING INSIDE (test data)")
    print("=" * 60)
    df_inside = evaluate_loader(test_inside_loader, "inside_vernier")

    print("\nDone! Now run: python analyze_crowding.py")