# Iterative-Document-Representation-Learning-Towards-Summarization-with-Polishing
 This is the official codes for the paper: Iterative Document Representation Learning Towards Summarization with Polishing

### Requirements

---

* Python 2.7
* Tensorflow 1.3.0

### CNN/Daily Mail dataset

---

Preprocessed data can be found [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).

Use `story2TF.py` to transform data into `.bin` format and put in `tfrecord` directory.

### How to train

---

```python
python single.py --train_dir=train_dir --train_path=path_to_training_data --validation_path=path_to_validation_data
```

### How to test

---

```python
python test.py --train_dir=path_to_train_dir --test_path=test_dir
```





