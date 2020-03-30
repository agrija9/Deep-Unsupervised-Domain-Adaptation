# Deep-Unsupervised-Domain-Adaptation

---

Pytorch implementation and performance evaluation of four domain adaptation techniques based on neural networks: DeepCORAL, DDC, CDAN and CDAN+E.

**Abstract**

> It has been well proved that deep networks are efficient at extracting features from a given (source) labeled dataset.
However, it is not always the case that they can generalize well to other (target) datasets which very often have a different underlying distribution. In this report, we evaluate four different domain adaptation techniques for image classification tasks: **Deep CORAL**, **Deep Domain Confusion (DDC)**, **Conditional Adversarial Domain Adaptation (CDAN)** and **CDAN with Entropy Conditioning (CDAN+E)**. 

> The selected domain adaptation techniques are unsupervised techniques where the target dataset will not carry any labels during training phase. The experiments are conducted on the office-31 dataset.

**Results**
---

Deep CORAL             |  DDC
:-------------------------:|:-------------------------:
![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/DEEP_CORAL_amazon_to_webcam_test_train_accuracies.jpg)  |  ![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/DDC_amazon_to_webcam_test_train_accuracies.jpg)

CDAN             |  CDAN+E
:-------------------------:|:-------------------------:
![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/CDAN_amazon_to_webcam_test_train_accuracies.png)  |  ![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/report/images/CDAN_E_amazon_to_webcam_test_train_accuracies.png)

**Training**
---

**References**
---
