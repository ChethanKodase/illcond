import sys


from datasets import load_dataset


'''
Download the imagenet subset of 10k images from hugging face (https://huggingface.co/datasets/Oztobuzz/ImageNet_10k/tree/main/data ).

There are 4 files train-00000-of-00004.parquet, train-00001-of-00004.parquet, train-00002-of-00004.parquet, train-00003-of-00004.parquet

Make a directory called mae/imagenetparaquet and add these files .

Make a directory called mae/imagenetDataSubset and run the below code


conda activate mae5
python mae/datasetSaver.py

'''



sys.path.append('..')

from datasets import load_dataset

dataset = load_dataset("parquet", data_files="mae/imagenetparaquet/train-*.parquet", split="train")

#os.makedirs("images", exist_ok=True)

for i, item in enumerate(dataset):
    image = item["image"]
    image.save(f"mae/imagenetDataSubset/{i}.jpg")

