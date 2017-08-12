#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:12:36 2017

@author: Sergio Carrozzo
"""

import ContentClassifier;
import sklearn.datasets;

def main():        
    train_data = sklearn.datasets.fetch_20newsgroups(subset='train');
    print len(train_data.filenames);
    test_data = sklearn.datasets.fetch_20newsgroups(subset='test');
    print(len(test_data.filenames));

    classifier = ContentClassifier.ContentClassifier(train_data.data[1:500], 'english');
    classifier.build_model(10);
    
    print "Model score on training dataset: %s" % classifier.get_model_score();
    print "Training dataset classes: %s" % classifier.get_training_dataset_classes();
    print "Centroids: %s" % classifier.get_cluster_centers();
    print "Class of some test data: %s" % classifier.predict(test_data.data[1:2]);

    similar_contents = classifier.get_most_similar_contents("Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks.", 3);
    print "3 closest similar contents of 'Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks.': %s" % similar_contents;
    
    # Elbow method
    #classifier.find_best_num_classes(5, 10);


main();