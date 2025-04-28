import os

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

original_size = get_folder_size("./models/bert_finetuned")
quantized_size = get_folder_size("./models/bert_finetuned_quantized")

print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Size reduction: {((original_size - quantized_size) / original_size) * 100:.2f}%")
