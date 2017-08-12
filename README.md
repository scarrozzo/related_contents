# Related Contents Classifier
Machine learning classifier for related contents.

A machine learning tool that allows to find related contents using k-means and the vectorizer lib of scikit (http://scikit-learn.org/).
It converts automatically a dataset of contents (strings) to a dataset of floating numbers using stemming, TF-IDF (term frequency – inverse document frequency) and removing the stop words of the language passed. Only English and Italian languages are supported for now.

Scikit (http://scikit-learn.org/stable/) and nltk (http://www.nltk.org/) are required.
See 'Example.py' for an example of usage.
