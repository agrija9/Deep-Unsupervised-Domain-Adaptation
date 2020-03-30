# Deep-Unsupervised-Domain-Adaptation

---

Pytorch implementation and performance evaluation of four deep neural network based domain adaptation techniques based on: DeepCORAL, DDC, CDAN and CDAN+E.

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

| A &#8594 W   | A &#8594 D     | W &#8594 A    | W &#8594 D  | D &#8594 A | D &#8594 W |
|  :---:       |     :---:      |    :---:      |  :---:      | :---:      | :---:      |   
| git status   | git status     | git status    | git         |  git       | git        |
| git diff     | git diff       | git diff      |             |            |            |


**Training**
---

**References**
---
