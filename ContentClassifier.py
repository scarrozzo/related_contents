#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:38:06 2017

@author: Sergio Carrozzo
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial import distance
import nltk
import pylab as plt

#==============================================================================
# Class that allows to create a bag of words feature vector from a string
#==============================================================================
class StemmedTfidfVectorizer(TfidfVectorizer):
     languages = {
                 'english': 'english',
                 'italian': [ 'a', 'adesso', 'ai', 'al', 'alla', 'allo', 'allora', 'altre', 'altri', 'altro', 'anche', 'ancora', 'avere', 'aveva', 'avevano', 'ben', 'buono', 'che', 'chi', 'cinque', 'comprare', 'con', 'consecutivi', 'consecutivo', 'cosa', 'cui', 'da', 'del', 'della', 'dello', 'dentro', 'deve', 'devo', 'di', 'doppio', 'due', 'e', 'ecco', 'fare', 'fine', 'fino', 'fra', 'gente', 'giu', 'ha', 'hai', 'hanno', 'ho', 'il', 'indietro	invece', 'io', 'la', 'lavoro', 'le', 'lei', 'lo', 'loro', 'lui', 'lungo', 'ma', 'me', 'meglio', 'molta', 'molti', 'molto', 'nei', 'nella', 'no', 'noi', 'nome', 'nostro', 'nove', 'nuovi', 'nuovo', 'o', 'oltre', 'ora', 'otto', 'peggio', 'pero', 'persone', 'piu', 'poco', 'primo', 'promesso', 'qua', 'quarto', 'quasi', 'quattro', 'quello', 'questo', 'qui', 'quindi', 'quinto', 'rispetto', 'sara', 'secondo', 'sei', 'sembra	sembrava', 'senza', 'sette', 'sia', 'siamo', 'siete', 'solo', 'sono', 'sopra', 'soprattutto', 'sotto', 'stati', 'stato', 'stesso', 'su', 'subito', 'sul', 'sulla', 'tanto', 'te', 'tempo', 'terzo', 'tra', 'tre', 'triplo', 'ultimo', 'un', 'una', 'uno', 'va', 'vai', 'voi', 'volte', 'vostro' ]
                };
    
     def __init__(self, language):
         if(language != 'english' and language != 'italian'):
             raise ValueError("Language not supported.");
             
         self.stemmer = nltk.stem.SnowballStemmer(language);   
         super(StemmedTfidfVectorizer, self).__init__(min_df=1, stop_words=self.languages[language],
                                        decode_error='ignore');

     def build_analyzer(self):
         analyzer = super(TfidfVectorizer, self).build_analyzer();
         return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc));


#==============================================================================
# Class that allows to build a classifier for similar contents/posts
#==============================================================================
class ContentClassifier:
    
    def __init__(self, contents, language):
        self.__vectorizer = StemmedTfidfVectorizer(language);
        self.__contents = contents;
        self.__dataset = self.__vectorizer.fit_transform(contents).todense();
        self.__num_samples, self.__num_features = self.__dataset.shape;
        
    # using k-means
    def build_model(self, num_classes, max_iter=300):
        self.__model = KMeans(n_clusters=num_classes, max_iter=max_iter);
        self.__model.fit(self.__dataset);
        
    # inverse of objective function of k-means
    def get_model_score(self):
        return self.__model.score(self.__dataset);
            
    # elbow method
    def find_best_num_classes(self, min_num_classes, max_num_classes):
        Ks = range(min_num_classes, max_num_classes+1);
        km = [KMeans(n_clusters=i) for i in Ks];
        score = [km[i].fit(self.__dataset).score(self.__dataset) for i in range(len(km))];
        plt.plot(Ks, score);
        plt.xlabel('num_classes');
        plt.ylabel('score');
        plt.title('Elbow method');
        plt.grid(True);
        
    def get_cluster_centers(self):
        return self.__model.cluster_centers_;
    
    def get_training_dataset_classes(self):
        return self.__model.labels_;
            
    def predict(self, contents):
        return self.__model.predict(self.__vectorizer.transform(contents));

    def get_most_similar_contents(self, content, num_contents=3):
        if(not isinstance(content, basestring)):
            raise TypeError("content must be a string");
        content_vectorized = self.__vectorizer.transform([content]).todense();
        distances = [];
        for i in range(0,len(self.__dataset)):
            distances.append(distance.euclidean(self.__dataset[i], content_vectorized));
        indexes = np.argsort(distances)[:min(num_contents, len(self.__contents))];
        return [{'content': self.__contents[i], 'score': distances[i]} for i in indexes];
        