import paddle
from PIL import Image
import os
import os.path as osp
from ppdiffusers import StableDiffusionImg2ImgPipeline
import csv

# load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)

ROOT_PATH = './data/Mini-ImageNet/'
prompt = "natural"

def generate_bad(setname):
    # read the splits
    csv_path = osp.join('splits_for_generate', setname + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    # generate samples based on each original image
    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(ROOT_PATH, setname, wnid, name)

        init_image = Image.open(path).convert("RGB")
        init_image = init_image.resize((84, 84))

        # generate "bad" samples
        images = pipe(prompt=prompt, image=init_image, strength=0.2, guidance_scale=7.5).images
        image = images[0]
        image = image.resize((84, 84))
        # save to "extra" folders
        filepath = osp.join(ROOT_PATH, setname, 'ext')
        if not osp.exists(filepath):
            os.mkdir(filepath)
        savepath = osp.join(filepath, 'ext_' + name)
        image.save(savepath, 'JPEG')


generate_bad('train')
generate_bad('test')
generate_bad('val')


