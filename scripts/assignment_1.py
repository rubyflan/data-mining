import sys, os, warnings
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn import metrics

from estimators import CustomNaiveBayesClassifier

import utils

warnings.filterwarnings("ignore")
def evaluate(model, data, label):
    pred = model.predict(data)
    
    acc = metrics.accuracy_score(label, pred)
    print(f'Accuracy: {100*acc:.1f}%')

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print('Enter filepaths of train and test datasets')
        sys.exit(0)

    elif(not os.path.exists(args[0])):
        print('Invalid filepath of the train dataset')
        sys.exit(0)

    elif(not os.path.exists(args[1])):
        print('Invalid filepath of the test dataset')
        sys.exit(0)

    else:
        train_data, train_labels  = utils.read_data(args[0])
        test_data, test_labels = utils.read_data(args[1])

        clf_pipeline = Pipeline([
            ('classifier', DummyClassifier(strategy="most_frequent")),
        ])
        clf_pipeline.fit(train_data, train_labels)
        evaluate(clf_pipeline, test_data, test_labels)

        clf_pipeline = Pipeline([
            ('vectorizer', CountVectorizer(analyzer='word')),
            ('classifier', MultinomialNB()),
        ])
        clf_pipeline.fit(train_data, train_labels)
        evaluate(clf_pipeline, test_data, test_labels)

        clf_pipeline.set_params(classifier = CustomNaiveBayesClassifier())
        clf_pipeline.fit(train_data, train_labels)
        evaluate(clf_pipeline, test_data, test_labels)
        
        # YOUR CODE HERE

if __name__ == "__main__":
    main()
