import paddle
from PIL import Image
import os
import os.path as osp
from ppdiffusers import StableDiffusionImg2ImgPipeline
import csv

# load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)

ROOT_PATH = './materials/'
prompt = "natural"


def generate_good(setname):
    # read the splits
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    # generate samples based on each original image
    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(ROOT_PATH, 'images', wnid, name)

        init_image = Image.open(path).convert("RGB")
        init_image = init_image.resize((84, 84))

        # generate "good" samples
        images = pipe(prompt=prompt, image=init_image, strength=0.05, guidance_scale=7.5).images
        image = images[0]
        image = image.resize((84, 84))

        # save to the original folder
        savepath = osp.join(ROOT_PATH, 'images', wnid, 'edi_' + name)
        image.save(savepath, 'JPEG')


def generate_bad(setname):
    # read the splits
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    # generate samples based on each original image
    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(ROOT_PATH, 'images', wnid, name)

        init_image = Image.open(path).convert("RGB")
        init_image = init_image.resize((84, 84))

        # generate "bad" samples
        images = pipe(prompt=prompt, image=init_image, strength=0.2, guidance_scale=7.5).images
        image = images[0]
        image = image.resize((84, 84))
        # save to "extra" folders
        filepath = osp.join(ROOT_PATH, 'images', 'ext_'+ wnid)
        if not osp.exists(filepath):
            os.mkdir(filepath)
        savepath = osp.join(filepath, 'ext_' + name)
        image.save(savepath, 'JPEG')


def update_splits(setname):
    input_file = osp.join(ROOT_PATH,setname + ".csv")
    output_file = "output.csv"

    # save labels to the generated images
    with open(input_file, "r") as in_file, open(output_file, "w", newline='') as out_file:
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)

        next(reader)

        for row in reader:
            edi_new_row = ["edi_" + str(row[0])] + row[1:]
            writer.writerow(edi_new_row)
            ext_new_row = ["ext_" + str(row[0])] + ["ext_" + str(row[1])]
            writer.writerow(ext_new_row)

    input_file = "output.csv"
    output_file = osp.join(ROOT_PATH,setname + ".csv")

    # add new labels to split files
    with open(input_file, "r") as in_file, open(output_file, "a", newline='') as out_file:
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)

        for row in reader:
            writer.writerow(row)
    os.remove("output.csv")


# generate_good('train')
# generate_good('test')
# generate_good('val')

generate_bad('train')
generate_bad('test')
generate_bad('val')

update_splits('train')
update_splits('test')
update_splits('val')


