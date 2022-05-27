# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
#hide
from fastbook import *

from fastai.vision.all import *
path = untar_data(URLs.PETS)

#hide
Path.BASE_PATH = path

fname = (path/"images").ls()[0]
fname

# +
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=setup_aug_tfms([
                     Rotate(draw=30, p=1, size=224), 
                     Zoom(draw=1.2, p=1., size=224),
                     Warp(draw_x=-0.2, draw_y=0.2, p=1., size=224)])
                 )

dls = pets.dataloaders(path/"images")
#pets.summary(path/"images")

# +

dls.train.show_batch(nrows=2, ncols=5)

# +

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
# -

sample = (path/'images').ls()[1,10,11]
sample, learn.predict(sample[1])
