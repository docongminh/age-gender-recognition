# End-To-End guideline this project
Basically, This project based-on Inception-Architecture to build model for Age Estimation and Gender Recognition Problem
# Project Structure
- Project: 
	- ```inceptionv4.py```        # building model architecture
	- ```training.py```		# Run training model
	- ```finetuning.py```		# Tuning model with new dataset
	- ```data_generator.py```	# Data generate for data augmentation
	- ```demo.py```		# Demo project with model trained
	- ```model_v4```		# Save model trained consists: weights and architecture
- Main library
```
	- Dlib
	- Keras==2.3.1
	- Opencv-python3
	- Tensorflow==1.14
```
- Demo
```
	python demo.py
```
- Default model in [model_v4](https://github.com/docongminh/joint-multi-task-age-estimation-gender-recognition-using-cnn/tree/master/model_v4)
```
	Face shape : 160 x 160 x 3
```
	
