#!/bin/sh

# imports and downloads the kaggle anime dataset
# remember to have your ~/.kaggle/kaggle.json
kaggle datasets download splcher/animefacedataset
unzip 'animefacedataset.zip'
mv images 
dataset
