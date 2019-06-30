# fakeNewsDetectionML

## **Introduction**
Recently, fake news has been a hot debated topic. Albeit not being a new phenomenon, a great number of people were brought to an awareness of fake news during the period of the 2016 US presidential election amongst the masses of fake news and clickbait about the two candidates. Not only an epidemic in America, it is everywhere within the Internet’s grasp. Just last year in India, a viral video of a child being kidnapped, later proven fabricated, was widespread throughout WhatsApp, instigating attacks on strangers, eventually leading to multiple instances of homicide. Online news disseminates quickly and efficiently, this case showing just how dangerous such a combination could be. These phenomenon makes me wonder, what can I do to help alleviate the situation using my profession? 

For my final year project, I decided to build a program that will be able to score an article based on its training data. Generally, higher score indicates that the article is more reliable. 

## **Problem Definition**
A news article is considered fake regarding the definition proposed by Claire Wardle in [Fake news. It's complicated](https://firstdraftnews.org/fake-news-complicated/) and aim to differentiate the veracity of an article by a content-based approach. The “content” refers to the text of the article. The investigation of other types of content such as audio, image, video, etc. is reserved for future work.

I define the problem as a binary classification problem, where an article is either classified as real or fake. Henceforth, given a corpus of labeled news articles with two classes: fake or real, the aim is to predict the class labels of unseen articles using a supervised learning approach.

## **Dataset Description**
Since I decided on a content-based and supervised learning method, all of our datasets must be labeled and have article text. These criteria make our search for datasets much harder since labeled data on this particular topic is difficult to find and copyright is also a big issue. I found three eligible datasets that are publicly available:

| Datasets	| Fake_or_real_news	| Liar	| Kaggle |
| --- | --- | --- | --- |
| Fake	| 3151	| 6889	| 7402 |
| Real	| 3159	| 8469	| 7303 |
| Total	| 6310	| 15358	| 14709 |

fake_or_real_news dataset is being used as the baseline dataset because:
(i) LIAR dataset is mostly compiled with short statements which is not the ideal training sample 
(ii) although Kaggle dataset’s contents fit our requirements and is significantly larger in size than the main dataset, the provider of the dataset does not give a detailed description of how they collect and label the dataset.

## **Text Preprocessing (preprocessing_text.py)**
The dataset is imported into program using pandas. The raw data contains a lot of punctuations and white spaces that are no help in achieving the task. The data need to be cleaned before feeding it into the classifier for a more meaningful outcome. Therefore, some preprocessing was being performed such as lowercasing the data, remove the stopwords, punctuation and whitespaces, and lemmatize the data using NLTK library to structure the data. Later on, the data is splitted into training and testing set for the next stage.

## **Feature Extraction (feature_extractor.py)**
In this stage, the training sets are being fed into count_vectorizer and TF_IDF_vectorizer using scikit-learn to be tokenized and extract features out of the corpus. Both vectorizers were applied same parameters: ignore words that reoccur in more than 50% of the document and others reoccurring less than 3% of the document, with the exception that TF_IDF has an additional parameter for unigrams and bigrams.

## **Modeling (classifier.py)**
I'm curious which classification algorithm works better in such case, so using the baseline datasets, I tested it on 8 classifiers scikit-learn has to offer: Naïve Bayes, Logistic Regression, Support Vector Machine (SVM), Decision Tree, Random Forest, Gradient Boosting Tree, Stochastic Gradient Descent (SGD), and a Multi-layer Perceptron classifier. The results are shown as below:

![alt text](C:\Users\sone_\Desktop\Project,Paper-related\result.png "result")

The application is hosted on [sushi.pythonanywhere.com](http://sushi.pythonanywhere.com). Feel free to try it out!


