# Machine Learning Homework 2 - Image Classification

This repository contains the code for the second homework of the Machine Learning course at the University of La Sapienza, Rome.

# Task description: Monkey Species Classification

As this homework is about image classification, I decided to use the [10 Monkey Species](https://www.kaggle.com/slothkong/10-monkey-species) dataset from Kaggle. The dataset contains 1400 images of 10 different monkey species, divided as in the following table:

| Label | Latin Name            | Common Name               | Train Images | Validation Images | Total Images |
| ----- | --------------------- | ------------------------- | ------------ | ----------------- | ------------ |
| n0    | alouatta_palliata     | mantled_howler            | 131          | 26                | 157          |
| n1    | erythrocebus_patas    | patas_monkey              | 139          | 28                | 167          |
| n2    | cacajao_calvus        | bald_uakari               | 137          | 27                | 164          |
| n3    | macaca_fuscata        | japanese_macaque          | 152          | 30                | 182          |
| n4    | cebuella_pygmea       | pygmy_marmoset            | 131          | 26                | 157          |
| n5    | cebus_capucinus       | white_headed_capuchin     | 141          | 28                | 169          |
| n6    | mico_argentatus       | silvery_marmoset          | 132          | 26                | 158          |
| n7    | saimiri_sciureus      | common_squirrel_monkey    | 142          | 28                | 170          |
| n8    | aotus_nigriceps       | black_headed_night_monkey | 133          | 27                | 160          |
| n9    | trachypithecus_johnii | nilgiri_langur            | 132          | 26                | 158          |

# Folder structure

```
project
│   main.py
│   ...   
│
└───data
│   └───training
│   |   └───n0
│   |   |   image1.jpg
│   |   |   ...
|   |   |
│   |   └───...
│   |   |
│   |   └───n9
│   |   │   image1.jpg
│   |   │   ...
|   |
│   └───validation
│   |   └───n0
│   |   |   image1.jpg
│   |   |   ...
|   |   |
│   |   └───...
│   |   |
│   |   └───n9
│   |   |   image1.jpg
│   |   |   ...
```


More details about the assignment can be found in the [report](report/main.pdf) PDF file.
