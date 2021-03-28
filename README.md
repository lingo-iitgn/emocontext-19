# emocontext-19
This repository contains our code for the paper titled **IIT Gandhinagar at SemEval-2019 Task 3: Contextual Emotion Detection Using Deep Learning**. Paper can be viewed at https://www.aclweb.org/anthology/papers/S/S19/S19-2039/

## Installation

Python 3.6.8 was used and to install all the dependencies do - 
```py
pip3 install -r requirements.txt
```
## Code

There are 2 broad approaches as mentioned in the paper
1. Non Deep Learning Approach

    * SVM
    * Logistic Regression

2. Deep Learning Approach

    * CNN
    * LSTM-1
    * LSTM-2

## Data
Dataset is explained in [this](./data/README.md) seperate README.

## Cite
If you use this in your work considering citing:

```
@inproceedings{pamnani-etal-2019-iit,
    title = "{IIT} {G}andhinagar at {S}em{E}val-2019 Task 3: Contextual Emotion Detection Using Deep Learning",
    author = "Pamnani, Arik  and
      Goel, Rajat  and
      Choudhari, Jayesh  and
      Singh, Mayank",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2039",
    doi = "10.18653/v1/S19-2039",
    pages = "236--240",
    abstract = "Recent advancements in Internet and Mobile infrastructure have resulted in the development of faster and efficient platforms of communication. These platforms include speech, facial and text-based conversational mediums. Majority of these are text-based messaging platforms. Development of Chatbots that automatically understand latent emotions in the textual message is a challenging task. In this paper, we present an automatic emotion detection system that aims to detect the emotion of a person textually conversing with a chatbot. We explore deep learning techniques such as CNN and LSTM based neural networks and outperformed the baseline score by 14{\%}. The trained model and code are kept in public domain.",
}
```
