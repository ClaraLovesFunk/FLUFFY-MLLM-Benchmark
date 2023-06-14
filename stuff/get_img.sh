#!/bin/bash

# Download validation images
wget http://images.cocodataset.org/zips/val2014.zip -P datasets/coco2014
unzip datasets/coco2014/val2014.zip -d datasets/coco2014
rm datasets/coco2014/val2014.zip

# Download test images
wget http://images.cocodataset.org/zips/test2014.zip -P datasets/coco2014
unzip datasets/coco2014/test2014.zip -d datasets/coco2014
rm datasets/coco2014/test2014.zip