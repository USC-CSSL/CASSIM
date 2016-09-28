__author__ = 'reihane'
import sklearn.utils.linear_assignment_ as su
import numpy as np
import sys
import os
from nltk.parse import stanford
import nltk
from nltk.tree import ParentedTree
from zss import simple_distance, Node
import random
numnodes =0

class Cassim:
    def __init__(self):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        os.environ['STANFORD_PARSER'] = 'jars/stanford-parser.jar'
        os.environ['STANFORD_MODELS'] = 'jars/stanford-parser-3.5.2-models.jar'
        self.parser = stanford.StanfordParser(model_path="jars/englishPCFG.ser.gz")

    def convert_mytree(self, nltktree,pnode):
        global numnodes
        for node in nltktree:
            numnodes+=1
            if type(node) is nltk.ParentedTree:
                tempnode = Node(node.label())
                pnode.addkid(tempnode)
                self.convert_mytree(node,tempnode)
        return pnode

    def minweight_edit_distance(self, doc1, doc2, average):
        global numnodes
        doc1sents = self.sent_detector.tokenize(doc1.strip())
        doc2sents = self.sent_detector.tokenize(doc2.strip())
        doc1parsed = self.parser.raw_parse_sents((doc1sents))
        doc2parsed = self.parser.raw_parse_sents((doc2sents))
        costMatrix = []
        doc1parsed = list(doc1parsed)
        for i in range(len(doc1parsed)):
            doc1parsed[i] = list(doc1parsed[i])[0]
        doc2parsed = list(doc2parsed)
        for i in range(len(doc2parsed)):
                doc2parsed[i] = list(doc2parsed[i])[0]
        for i in range(len(doc1parsed)):
            numnodes = 0
            sentencedoc1 = ParentedTree.convert(doc1parsed[i])
            tempnode = Node(sentencedoc1.root().label())
            new_sentencedoc1 = self.convert_mytree(sentencedoc1,tempnode)
            temp_costMatrix = []
            sen1nodes = numnodes
            for j in range(len(doc2parsed)):
                numnodes=0.0
                sentencedoc2 = ParentedTree.convert(doc2parsed[j])
                tempnode = Node(sentencedoc2.root().label())
                new_sentencedoc2 = self.convert_mytree(sentencedoc2,tempnode)
                ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                ED = ED / (numnodes + sen1nodes)
                temp_costMatrix.append(ED)
            costMatrix.append(temp_costMatrix)
        costMatrix = np.array(costMatrix)
        if average==True:
            return 1-np.mean(costMatrix)
        else:
            rownum= costMatrix.shape[0]
            colnum = costMatrix.shape[1]
            if rownum > colnum:
                costMatrixRandom = costMatrix[np.random.randint(rownum, size=colnum),:]
            else:
                costMatrixRandom = costMatrix[:,np.random.randint(colnum, size=rownum)]

            indexes = su.linear_assignment(costMatrix)
            total = 0
            minWeight = 0
            rowMarked = [0] * len(doc1parsed)
            colMarked = [0] * len(doc2parsed)
            for row, column in indexes:
                total += costMatrix[row][column]
                rowMarked[row] = 1
                colMarked [column] = 1
            minWeight = total

            for k in range(len(rowMarked)):
                if rowMarked[k]==0:
                    total+= np.min(costMatrix[k])
            for c in range(len(colMarked)):
                if colMarked[c]==0:
                    total+= np.min(costMatrix[:,c])
            maxlengraph = max(len(doc1parsed),len(doc2parsed))
            minlengraph = min(len(doc1parsed),len(doc2parsed))

            indexes = su.linear_assignment(costMatrixRandom)
            randtotal = 0
            for row, column in indexes:
                randtotal +=costMatrixRandom[row][column]
            lengraph = costMatrixRandom.shape[0]

            return 1-total/maxlengraph#, minWeight/minlengraph, randtotal/lengraph
