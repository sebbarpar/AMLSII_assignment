# AMLSII_assignment
Image classification task with 6 classes. Dataset link: https://www.kaggle.com/puneet6060/intel-image-classification
Dataset is too big to upload. To test code download dataset and insert in /dataset folder.
This project is separated in three parts:

Part 1: Base classifier.
Three architectures were tested. Only VGG will be used. The code for the other two is available as Jupyter notebook.

Part 2: Mixture of experts.
Three models are created: two experts and a two-class gate.

Part 3: Final architecture. 
Bringing all together, can be found in main.py.

How to execute:
1. Run VGG.py
2. Run Two_class.py
3. Run Nature.py and Man_made.py
4. Run main.py

The models trained in each part will be stored and imported to main.py to bring all together and create final model.
