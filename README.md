# Deep-Unsupervised-Domain-Adaptation

---

Pytorch implementation of four neural network based domain adaptation techniques: DeepCORAL, DDC, CDAN and CDAN+E. Evaluated on benchmark dataset Office31.

**Abstract**

> It has been well proved that deep networks are efficient at extracting features from a given (source) labeled dataset.
However, it is not always the case that they can generalize well to other (target) datasets which very often have a different underlying distribution. In this report, we evaluate four different domain adaptation techniques for image classification tasks: **Deep CORAL**, **Deep Domain Confusion (DDC)**, **Conditional Adversarial Domain Adaptation (CDAN)** and **CDAN with Entropy Conditioning (CDAN+E)**. The selected domain adaptation techniques are unsupervised techniques where the target dataset will not carry any labels during training phase. The experiments are conducted on the office-31 dataset.

**Results**
---

Accuracy performance on the Office31 dataset for the source and domain data distributions (with and without transfer losses).

Deep CORAL             |  DDC
:-------------------------:|:-------------------------:
![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/DEEP_CORAL_amazon_to_webcam_test_train_accuracies.jpg)  |  ![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/DDC_amazon_to_webcam_test_train_accuracies.jpg)

CDAN             |  CDAN+E
:-------------------------:|:-------------------------:
![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/CDAN_amazon_to_webcam_test_train_accuracies.png)  |  ![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/CDAN_E_amazon_to_webcam_test_train_accuracies.png)

Target accuracies for all six domain shifts in Office31 dataset (amazon, webcam and dslr)

| Method         | A &#8594; W   | A &#8594; D  | W &#8594; A    | W &#8594; D  | D &#8594; A    | D &#8594; W     |
| :---:          |  :---:        |     :---:    |    :---:       |  :---:       | :---:          | :---:           |   
| No Adaptaion   | 43.1 ± 2.5    | 49.2 ± 3.7   |   35.6 ± 0.6   |  94.2 ± 3.1  | 35.4 ± 0.7     |  90.9 ± 2.4     |   
| DeepCORAL      | **49.5 ± 2.7**| 40.0 ± 3.3   | **38.3 ± 0.4** | 74.4 ± 4.3   | **38.5 ± 1.5** | **89.1 ± 4.4**  |
| DDC            | 41.7 ± 9.1    | ---          | ---            | ---          | ---            | ---             |
| CDAN           | 44.9 ± 3.3    | 49.5 ± 4.6   | 34.8 ± 2.4     | 93.3 ± 3.4   | 32.9 ± 3.4     |  88.3 ± 3.8     |
| CDAN+E         | 48.7 ± 7.5    |**53.7 ± 4.7**| 35.3 ± 2.7     |**93.6 ± 3.4**| 33.9 ± 2.2     | 87.7 ± 4.0      |



**Training and inference**
---

To train the model in your computer you must download the [**Office31**](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view) dataset and put it in your data folder. 

Execute training of a method by going to its folder (e.g. DeepCORAL):

```
cd DeepCORAL/
python main.py --epochs 100 --batch_size_source 128 --batch_size_target 128 --name_source amazon --name_target webcam
```

**Loss and accuracy plots**
---

Once the model is trained, you can generate plots like the ones shown above by running:

```
cd DeepCORAL/
python plot_loss_acc.py --epochs 100 --batch_size_source 128 --batch_size_target 128 --name_source amazon --name_target webcam
```

The following is a list of the arguments the usuer can provide:

```--epochs``` number of training epochs
```--batch_size_source``` batch size of source data
```--batch_size_target``` batch size of target data
```--name_source``` name of source dataset
```--name_target``` name of source dataset
```--num_classes``` no. classes in dataset
```--load_model``` flag to load pretrained model (AlexNet by default)
```--adapt_domain``` bool argument to train with or without specific transfer loss

**References**
---

- [DeepCORAL](https://arxiv.org/abs/1607.01719) paper
- [DDC](https://arxiv.org/abs/1412.3474) paper
- [CDAN](https://arxiv.org/abs/1705.10667) paper
