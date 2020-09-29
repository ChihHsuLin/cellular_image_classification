Kaggle competition 2019: Recursion Cellular Image Classification
======================
This is the solution for Kaggle competition [Recursion Cellular Image Classification](http://kaggle.com/c/recursion-cellular-image-classification/) based on the efforts of [Tse-Ju Lin](https://github.com/lintseju) and [Chih-Hsu Jack Lin](https://www.linkedin.com/in/chihhsulin/).
We scored multiclass accuracy of 0.97757, ranked as 26/866 (top 3.0%) and won a silver medal in the end. To note, half of the teams ranked higher than us had more team members than us. We only had two persons and two GPUs and training this image dataset usually took almost a day. Therefore, we are pretty proud of what we have achieved.

Problem
----
Drug development is costly and can be a decade-long process.
In this competition, we were challenged to classify images of human cells (total 4 types) under one of 1,108 different genetic perturbations. By disentangling experimental noise from real biological signals, the interactions between drugs and human cells could be understood better. As a result, the drug discovery process can be improved and expedited.

Final solution
----
- Overall strategy: 
  1. Train one model with all data
  2. Fine tune the overall model by training data from only one cell type to generate 4 cell line-specific models
  3. Ensemble (taking the average of top predictions from various models with different validation data)

- Models:
  - [EfficientNet](https://arxiv.org/abs/1905.11946) b2, b3, b4
  - EfficientNet of the  [Mish](https://arxiv.org/abs/1908.08681) activation 

- Data processing
	- Split 512x512 image into four 256x256 images
- Training strategy
	- Treat the two images from different microscopic fields of the same sample id as two separate samples to double training sample size
	- Each sample randomly selects one of four 256x256 image
	- Data augmentation
		- Flip (vertical, horizontal,  vertical+horizontal)
		- Rotation (90, 180, 270 degree)
	- Early stopping by the loss of validation data
	- To increase the ensemble model diversity, we changed validation and training data for different models.
	- Only use the data from the same cell type as validation data during fine tuning
- Validation data selection
	- Split the data by cell type and experiment.

- Test strategy
	- Data augmentation
		- Flip (vertical, horizontal,  vertical+horizontal)
		- Rotation (90, 180, 270 degree)
	- Average the predictions from all replicates and augmented images as final prediction
- Others
	- U2OS cell type has the fewest data among 4 cell types and the cellular images look very different from others. Therefore, we calculated the metrics for each cell type and chose different models to build ensemble models for U2OS and 3 other cell types.

- Final submission selection
	- We chose the ensemble model of higher diversity. Even though we did not choose the submission of highest score on public leaderboard, one of the two submissions we selected was the one of highest score on private leaderboard.


Lessons
----
1. Different cell types have pretty distinct images so the cell type-specific models are necessary.
2. More training data and train-/test-time augmentation are usually helpful.
3. Splitting high-resolution images into smaller ones instead of shrinking images can achieve the training using more information without requiring more memory of GPU.
4. The more diverse ensemble models provide better generalizable predictions. This was proven by one of our submissions. Even it had higher scores on public leader board, it had lower scores on private leader board in the end, potentially due to its lower diversity of models.
5. Even with the same architecture, the same hyperparameter set and the same validation set, the mean ensemble still improved the performance! It demonstrates how huge the differences among the solutions which can be found by the the neural networks.
6. We tried [ArcFace](https://arxiv.org/abs/1801.07698) but couldn't make it work. After the competition ended, we found many top teams successfully used Arcface loss by different ways: (i) training the overall model using softmax and fine tuning using ArcFace loss; (ii) using softmax (*0.8) and ArcFace loss (*0.2) as the total loss; (iii) using other parameter sets of ArcFace that we didn't have time to explore.
7. Pseudo-labeling was shown helpful by many other top teams but we didn't have time to try it. Essentially, the idea is to add those test data with confident predictions into to the training process to increase the data used by the model. It is especially helpful for small datasets. Like this dataset has only ~15 training images per class per cell type on average.
8. Team work is the key. Finding a good team mate saves you lots of time and efforts.


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The package requirements of this proejct
    │
    └── src                <- Source code for use in this project.
        │
        ├── __init__.py    <- Makes src a Python module
        │
        ├── features       <- Scripts to preprocess the images before training
        │
        └── models         <- Scripts to train models and then use trained models to make
                              predictions

Steps to run
------------
1. Install required packages. `pip3 install -r requirements.txt`
2. Preprocess the data using `src/features/preprocess.py`
3. Train the model using `python src/models/train.py -y src/models/config/train.yml`
4. Fine tune the model using `python src/models/fine_tune.py -y src/models/config/fine_tune.yml`
5. Make predictions for test data using `python src/models/inference.py -y src/models/config/test.yml`
6. Make ensemble predictions using `python src/models/ensemble.py -y src/models/config/ensemble.yml`

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
