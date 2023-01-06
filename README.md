# Machine Learning Homework 2 - Image Classification

This repository contains the code for the second homework of the Machine Learning course at the University of La Sapienza, Rome.

# Task description: Monkey Species Classification

As this homework is about image classification, I decided to use the [10 Monkey Species](https://www.kaggle.com/slothkong/10-monkey-species) dataset from Kaggle. The dataset contains 1400 images of 10 different monkey species, divided as in the following table:

| Label | Latin Name            | Common Name               | Train Images | Validation Images | Total Images |
| ----- | --------------------- | ------------------------- | ------------ | ----------------- | ------------ |
| n0    | Alouatta Palliata     | Mantled Howler            | 131          | 26                | 157          |
| n1    | Erythrocebus Patas    | Patas Monkey              | 139          | 28                | 167          |
| n2    | Cacajao Calvus        | Bald Uakari               | 137          | 27                | 164          |
| n3    | Macaca Fuscata        | Japanese Macaque          | 152          | 30                | 182          |
| n4    | Cebuella Pygmea       | Pygmy Marmoset            | 131          | 26                | 157          |
| n5    | Cebus Capucinus       | White Headed Capuchin     | 141          | 28                | 169          |
| n6    | Mico Argentatus       | Silvery Marmoset          | 132          | 26                | 158          |
| n7    | Saimiri Sciureus      | Common Squirrel Monkey    | 142          | 28                | 170          |
| n8    | Aotus Nigriceps       | Black Headed Night Monkey | 133          | 27                | 160          |
| n9    | Trachypithecus Johnii | Nilgiri Langur            | 132          | 26                | 158          |

# Folder structure

```
project folder
│   main.py
│   ...   
│
└───data
|   |   monkey_labels.txt
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
