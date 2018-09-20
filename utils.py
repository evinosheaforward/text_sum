import string
import re

import gensim.models as g

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

import numpy

import matplotlib

"""Utilies for text summarization"""



###############################################################################
#python example to train doc2vec model (with or without pre-trained word embeddings)

def train_doc2vec(pretrained_emb=None, train_corpus=None, saved_path=None):
    #doc2vec parameters
    vector_size = 300
    window_size = 15
    min_count = 1
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 100
    dm = 0 #0 = dbow; 1 = dmpv
    worker_count = 1 #number of parallel processes

    #pretrained word embeddings
    if pretrained_emb is None:
        pretrained_emb = "/home/eoshea/sflintro/doc2vec/toy_data/pretrained_word_embeddings.txt" #None if use without pretrained embeddings

    #input corpus
    if train_corpus is None:
        train_corpus = "/home/eoshea/sflintro/doc2vec/toy_data/train_docs.txt"

    #output model
    if saved_path is None:
        saved_path = "/home/eoshea/sflintro/doc2vec/toy_data/model.bin"

    #train doc2vec model
    docs = g.doc2vec.TaggedLineDocument(train_corpus)
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)

    #save model
    model.save(saved_path)

    return model

def load_doc2vec(path=None):
    if path is None:
        path = "/home/eoshea/sflintro/doc2vec/toy_data/model.bin"

    return g.Doc2Vec.load(path)

###############################################################################

def read_data(doc):
    """Parse doc to list of sentances"""
    begin = 0
    while begin != -1:
        begin = doc.find('writes:\n')
        if begin != -1:
            doc = doc[  doc.find('writes:\n') + 8 :]

    doc = re.sub(r'\n', '', doc)

    # regex = r'[' + string.punctuation + r']'
    # doc = re.sub(regex, '', doc)
    
    special = ''
    for i in string.punctuation:
        if i not in ".?!'\",:":
            special = special + i

    #print(special)
    regex = r'[' + special + r']re+'
    doc = re.sub(regex, '', doc)

    sentances = [i.strip() for i in re.split(r'[{0}]'.format('.!?'), doc)]

    return sentances


def tokenizer(sentance):  
    ne_tree = ne_chunk(pos_tag(word_tokenize(sentance.encode('utf-8', 'ignore'))))

    iob_tagged = tree2conlltags(ne_tree)

    words = [i[0] for i in iob_tagged]
    return words 


class TextSummary(object):
    """A class for text summarization. The class is initialized with parameters that
    determine the method for sumarization, summary is obtained by the extract_summary 
    function."""

    def __init__(self, model, num_out, vectorizer=None, method='nns'):
        self.model = model
        self.vectorizer = vectorizer
        self.num_out = num_out
        self.method = method
        self.start_alpha = 0.01
        self.infer_epoch = 1000

    def plot_nns(self):
        ranked_sentances = sorted([self.num_nouns(self.document[i]) \
                        for i in range(len(self.document))], reverse=True)

        matplotlib.pyplot.plot(range(len(ranked_sentances)), ranked_sentances)
        matplotlib.pyplot.show()

    def plot_dots(self):
        ranked_sentances = sorted([self.sentance_importance(self.document[i]) \
                        for i in range(len(self.document))], reverse=True)

        tot = sum(ranked_sentances)
        normed_sents = [i/tot for i in ranked_sentances]

        matplotlib.pyplot.plot(range(len(normed_sents)), normed_sents)
        matplotlib.pyplot.show()

    def extract_summary(self, document):
        """Do the process of extracting the test summary"""
        self.document = list(set(document))

        #use the number of nouns to rank sentance importance
        if self.method is 'nns':
            if self.vectorizer is None:
                compare_sentances = self.compare_sentances_pretrained
            else:
                compare_sentances = self.compare_sentances_vectorizer

            return self.most_information_nns(compare_sentances)

        #find sentance importance by comparing it to all other sentances
        elif self.method is 'similarity':
            if self.vectorizer is None:
                self.sentance_importance = self.sentance_similarity_pretrained
            else:
                self.sentance_importance = self.sentance_similarity_vectorizer
            
            return self.most_information_similarity()

    def most_information_similarity(self):
        """Extract the sentances with the most information based on each
        sentances's overall similarity to all other sentances"""

        #sort these by importance score
        ranked_sentances = sorted([(self.document[i], self.sentance_importance(self.document[i]), i) \
                        for i in range(len(self.document))], key=lambda x: x[1], reverse=True)

        #return the sentances in the order of appearance, only the top num_out
        return [i[0] for i in sorted(ranked_sentances[:self.num_out], key=lambda x: x[2])]


    def most_information_nns(self, compare_sentances):
        """Extract the sentances with the most information
        """

        #sort these by importance score
        ranked_sentances = sorted([(self.document[i], self.num_nouns(self.document[i]), i) \
                        for i in range(len(self.document))], key=lambda x: x[1], reverse=True)

        comp = []
        for sent, importance, order in ranked_sentances:
            for scomp, importancecomp, ordercomp in ranked_sentances:
                if sent != scomp and importance > importancecomp:
                    comp.append( [scomp,  compare_sentances(sent, scomp)])
                else: 
                    continue

        combed = self.comb_information(ranked_sentances, comp)

        return [i[0] for i in sorted(combed[:self.num_out], key=lambda x: x[2])]

    def num_nouns(self, sentance):  
        """Return number of nouns in sentance reguardless of type"""
        ne_tree = ne_chunk(pos_tag(word_tokenize(sentance)))#.encode('utf-8', 'ignore')
        iob_tagged = tree2conlltags(ne_tree)

        nn = len([i for i in iob_tagged if i[1].startswith('N')])

        return nn

    def sentance_similarity_pretrained(self, sentance):
        """calculate the importance score for a sentance by comparing its similarity to the whole document"""

        return sum([numpy.dot(
            self.model.infer_vector(sentance, alpha=self.start_alpha, steps=self.infer_epoch), 
            self.model.infer_vector(comp_sent, alpha=self.start_alpha, steps=self.infer_epoch))
            for comp_sent in self.document if sentance != comp_sent])

    def sentance_similarity_vectorizer(self, sentance):
        """calculate the importance score for a sentance by comparing its similarity to the whole document"""

        dots = []
        for comp_sent in self.document:
            if sentance != comp_sent:
                svects = self.model.transform(self.vectorizer.transform([sentance, comp_sent]))
                dots.append(numpy.dot(svects[0], svects[1])) 

        return sum(dots)
            

    def compare_sentances_pretrained(self, s1, s2):
        """comapre sentances using a model passed"""

        return numpy.dot(self.model.infer_vector(s1, alpha=self.start_alpha, steps=self.infer_epoch), 
                         self.model.infer_vector(s2, alpha=self.start_alpha, steps=self.infer_epoch))

    def compare_sentances_vectorizer(self, s1, s2):
        """compare two sentances given there is a vectorizer and a topic model being used together""" 

        svect = self.model.transform(self.vectorizer.transform([s1, s2]))
        return numpy.dot(svect[0], svect[1])

    @staticmethod
    def comb_information(nns, comparison, num_out=10):
        """remove similar sentances (for importance ranked by number of nouns""" 
        dots = sorted(comparison, key=lambda x: x[1], reverse=True)

        for dot in dots[:num_out]:
            if len(nns) <= num_out:
                break

            to_rm = []
            for i in range(len(nns)):
                if nns[i][0] == dot[0]:
                    to_rm.append(nns[i])

            if len(to_rm) > 1:
                raise ValueError("Two exact same sentances found in data.")
            if to_rm:
                nns.remove(to_rm[0])

        return nns