import paddle
from PIL import Image
import os
import os.path as osp
from ppdiffusers import StableDiffusionImg2ImgPipeline
import csv
import pandas as pd
import random

load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)

ROOT_PATH = './data/mini-imagenet/'
prompt = "natural"

def random_pick(setname):
    df = pd.read_csv(osp.join('split','miniImageNet' ,setname + '.csv'))
    file_types = df['label'].unique()

    result_df = pd.DataFrame(columns=['filename', 'label'])

    # randomly selected 5 images per class
    for file_type in file_types:
        file_names = df[df['label'] == file_type]['filename'].tolist()
        selected_files = random.sample(file_names, k=5)
        for file_name in selected_files:
            result_df = result_df.append({'filename': file_name, 'label': file_type}, ignore_index=True)

    # save the results of the random selection
    result_df.to_csv(osp.join('splits_for_generation' ,setname + '.csv'), index=False)

def generate_bad(setname):
    # read the splits
    csv_path = osp.join('splits_for_generation', setname + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    # generate samples based on each original image
    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(ROOT_PATH, 'images', name)

        init_image = Image.open(path).convert("RGB")
        init_image = init_image.resize((224, 224))

        # generate "bad" samples
        images = pipe(prompt=prompt, image=init_image, strength=0.3, guidance_scale=7.5).images
        image = images[0]
        image = image.resize((224, 224))
        # save to "extra" folders
        filepath = osp.join(ROOT_PATH, 'images')
        savepath = osp.join(filepath, 'ext_' + name)
        image.save(savepath, 'JPEG')


def update_splits(setname):
    input_file = osp.join('splits_for_generation', setname + ".csv")
    output_file = osp.join('split', 'miniImageNet', setname + '.csv')

    # add new labels to split files
    with open(input_file, "r") as in_file, open(output_file, "a", newline='') as out_file:
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)

        next(reader)
        for row in reader:
            new_row = ["ext_" + str(row[0])] + ["ext"]
            writer.writerow(new_row)

random_pick('train')
random_pick('test')
random_pick('val')

generate_bad('train')
generate_bad('test')
generate_bad('val')

update_splits('train')
update_splits('test')
update_splits('val')


