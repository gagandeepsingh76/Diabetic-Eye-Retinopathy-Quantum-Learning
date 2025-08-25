import os

def check_dataset_balance(base_path):
    splits = ['train', 'val']
    for split in splits:
        split_path = os.path.join(base_path, split)
        print(f"\nChecking '{split}' dataset:")

        class_counts = {}
        total = 0

        # Count images per class
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                class_counts[cls] = count
                total += count

        # Print percentages
        for cls, count in class_counts.items():
            percentage = 100 * count / total
            print(f"  {cls}: {count} images ({percentage:.2f}%)")

        # Quick balance check
        if max(class_counts.values()) - min(class_counts.values()) < 0.1 * total:
            print("  ✅ Balanced")
        else:
            print("  ❌ Imbalanced")

# Run
check_dataset_balance("im1_balanced")
