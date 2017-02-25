__author__ = 'reihane'
import sklearn.utils.linear_assignment_ as su
import numpy as np
import sys
import os
from nltk.parse import stanford
import nltk
from nltk.tree import ParentedTree
from zss import simple_distance, Node
numnodes =0
from collections import OrderedDict
class Cassim:
    def __init__(self, swbd=False):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        os.environ['STANFORD_PARSER'] = 'jars/stanford-parser.jar'
        os.environ['STANFORD_MODELS'] = 'jars/stanford-parser-3.5.2-models.jar'
        if swbd == False:
            self.parser = stanford.StanfordParser(model_path="jars/englishPCFG.ser.gz")
        else:
            self.parser = stanford.StanfordParser(model_path="jars/englishPCFG_swbd.ser.gz")

    def convert_mytree(self, nltktree,pnode):
        global numnodes
        for node in nltktree:
            numnodes+=1
            if type(node) is nltk.ParentedTree:
                tempnode = Node(node.label())
                pnode.addkid(tempnode)
                self.convert_mytree(node,tempnode)
        return pnode

    def syntax_similarity_two_documents(self, doc1, doc2, average=False): #syntax similarity of two single documents
        global numnodes
        doc1sents = self.sent_detector.tokenize(doc1.strip())
        doc2sents = self.sent_detector.tokenize(doc2.strip())
        for s in doc1sents: # to handle unusual long sentences.
            if len(s.split())>100:
                return "NA"
        for s in doc2sents:
            if len(s.split())>100:
                return "NA"
        try: #to handle parse errors. Parser errors might happen in cases where there is an unsuall long word in the sentence.
            doc1parsed = self.parser.raw_parse_sents((doc1sents))
            doc2parsed = self.parser.raw_parse_sents((doc2sents))
        except Exception as e:
            sys.stderr.write(str(e))
            return "NA"
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
            indexes = su.linear_assignment(costMatrix)
            total = 0
            rowMarked = [0] * len(doc1parsed)
            colMarked = [0] * len(doc2parsed)
            for row, column in indexes:
                total += costMatrix[row][column]
                rowMarked[row] = 1
                colMarked [column] = 1
            for k in range(len(rowMarked)):
                if rowMarked[k]==0:
                    total+= np.min(costMatrix[k])
            for c in range(len(colMarked)):
                if colMarked[c]==0:
                    total+= np.min(costMatrix[:,c])
            maxlengraph = max(len(doc1parsed),len(doc2parsed))
            return 1-(total/maxlengraph)
    def syntax_similarity_two_lists(self, documents1, documents2, average = False): # synax similarity of two lists of documents
        global numnodes
        documents1parsed = []
        documents2parsed = []

        for d1 in range(len(documents1)):
            # print d1
            tempsents = (self.sent_detector.tokenize(documents1[d1].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents1parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents1parsed.append(list(temp))
        for d2 in range(len(documents2)):
            # print d2
            tempsents = (self.sent_detector.tokenize(documents2[d2].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents2parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents2parsed.append(list(temp))
        results ={}
        for d1 in range(len(documents1parsed)):
            # print d1
            for d2 in range(len(documents2parsed)):
                # print d1,d2
                if documents1parsed[d1]=="NA" or documents2parsed[d2] =="NA":
                    # print "skipped"
                    continue
                costMatrix = []
                for i in range(len(documents1parsed[d1])):
                    numnodes = 0
                    tempnode = Node(documents1parsed[d1][i].root().label())
                    new_sentencedoc1 = self.convert_mytree(documents1parsed[d1][i],tempnode)
                    temp_costMatrix = []
                    sen1nodes = numnodes
                    for j in range(len(documents2parsed[d2])):
                        numnodes=0.0
                        tempnode = Node(documents2parsed[d2][j].root().label())
                        new_sentencedoc2 = self.convert_mytree(documents2parsed[d2][j],tempnode)
                        ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                        ED = ED / (numnodes + sen1nodes)
                        temp_costMatrix.append(ED)
                    costMatrix.append(temp_costMatrix)
                costMatrix = np.array(costMatrix)
                if average==True:
                    return 1-np.mean(costMatrix)
                else:
                    indexes = su.linear_assignment(costMatrix)
                    total = 0
                    rowMarked = [0] * len(documents1parsed[d1])
                    colMarked = [0] * len(documents2parsed[d2])
                    for row, column in indexes:
                        total += costMatrix[row][column]
                        rowMarked[row] = 1
                        colMarked [column] = 1
                    for k in range(len(rowMarked)):
                        if rowMarked[k]==0:
                            total+= np.min(costMatrix[k])
                    for c in range(len(colMarked)):
                        if colMarked[c]==0:
                            total+= np.min(costMatrix[:,c])
                    maxlengraph = max(len(documents1parsed[d1]),len(documents2parsed[d2]))
                    results[(d1,d2)] = 1-total/maxlengraph
        return results

    def syntax_similarity_conversation(self, documents1, average=False): #syntax similarity of each document with its before and after document
        global numnodes
        documents1parsed = []
        for d1 in range(len(documents1)):
            sys.stderr.write(str(d1)+"\n")
            # print documents1[d1]
            tempsents = (self.sent_detector.tokenize(documents1[d1].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents1parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents1parsed.append(list(temp))
        results = OrderedDict()
        for d1 in range(len(documents1parsed)):
            d2 = d1+1
            if d2 == len(documents1parsed):
                break
            if documents1parsed[d1] == "NA" or documents1parsed[d2]=="NA":
                continue
            costMatrix = []
            for i in range(len(documents1parsed[d1])):
                numnodes = 0
                tempnode = Node(documents1parsed[d1][i].root().label())
                new_sentencedoc1 = self.convert_mytree(documents1parsed[d1][i],tempnode)
                temp_costMatrix = []
                sen1nodes = numnodes
                for j in range(len(documents1parsed[d2])):
                    numnodes=0.0
                    tempnode = Node(documents1parsed[d2][j].root().label())
                    new_sentencedoc2 = self.convert_mytree(documents1parsed[d2][j],tempnode)
                    ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                    ED = ED / (numnodes + sen1nodes)
                    temp_costMatrix.append(ED)
                costMatrix.append(temp_costMatrix)
            costMatrix = np.array(costMatrix)
            if average==True:
                return 1-np.mean(costMatrix)
            else:
                indexes = su.linear_assignment(costMatrix)
                total = 0
                rowMarked = [0] * len(documents1parsed[d1])
                colMarked = [0] * len(documents1parsed[d2])
                for row, column in indexes:
                    total += costMatrix[row][column]
                    rowMarked[row] = 1
                    colMarked [column] = 1
                for k in range(len(rowMarked)):
                    if rowMarked[k]==0:
                        total+= np.min(costMatrix[k])
                for c in range(len(colMarked)):
                    if colMarked[c]==0:
                        total+= np.min(costMatrix[:,c])
                maxlengraph = max(len(documents1parsed[d1]),len(documents1parsed[d2]))
                results[(d1,d2)] = 1-total/maxlengraph#, minWeight/minlengraph, randtotal/lengraph
        return results

    def syntax_similarity_one_list(self, documents1, average): #syntax similarity of each document with all other documents
        global numnodes
        documents1parsed = []
        for d1 in range(len(documents1)):
            print d1
            tempsents = (self.sent_detector.tokenize(documents1[d1].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents1parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents1parsed.append(list(temp))
        results ={}
        for d1 in range(len(documents1parsed)):
            print d1
            for d2 in range(d1+1 , len(documents1parsed)):
                if documents1parsed[d1] == "NA" or documents1parsed[d2]=="NA":
                    continue
                costMatrix = []
                for i in range(len(documents1parsed[d1])):
                    numnodes = 0
                    tempnode = Node(documents1parsed[d1][i].root().label())
                    new_sentencedoc1 = self.convert_mytree(documents1parsed[d1][i],tempnode)
                    temp_costMatrix = []
                    sen1nodes = numnodes
                    for j in range(len(documents1parsed[d2])):
                        numnodes=0.0
                        tempnode = Node(documents1parsed[d2][j].root().label())
                        new_sentencedoc2 = self.convert_mytree(documents1parsed[d2][j],tempnode)
                        ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                        ED = ED / (numnodes + sen1nodes)
                        temp_costMatrix.append(ED)
                    costMatrix.append(temp_costMatrix)
                costMatrix = np.array(costMatrix)
                if average==True:
                    return 1-np.mean(costMatrix)
                else:
                    indexes = su.linear_assignment(costMatrix)
                    total = 0
                    rowMarked = [0] * len(documents1parsed[d1])
                    colMarked = [0] * len(documents1parsed[d2])
                    for row, column in indexes:
                        total += costMatrix[row][column]
                        rowMarked[row] = 1
                        colMarked [column] = 1
                    for k in range(len(rowMarked)):
                        if rowMarked[k]==0:
                            total+= np.min(costMatrix[k])
                    for c in range(len(colMarked)):
                        if colMarked[c]==0:
                            total+= np.min(costMatrix[:,c])
                    maxlengraph = max(len(documents1parsed[d1]),len(documents1parsed[d2]))
                    results[(d1,d2)] = 1-total/maxlengraph#, minWeight/minlengraph, randtotal/lengraph
        return results
