#dominhkha
import numpy as np 
import os
from collections import Counter,defaultdict
from functools import partial
cwd=os.getcwd()

def entropy(classes_probability):
    return -np.sum(p*np.log(p) for p in classes_probability if p)

def class_probabilities(labels):
    total=len(labels)
    return [count/total for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels=[label for _,label in labeled_data]
    probabilities=class_probabilities(labels)
    return entropy(probabilities)
def partition_entropy(subsets):
    total_count=np.sum(len(subset) for subset in subsets)
    return np.sum(data_entropy(subset)*len(subset)/total_count for subset in subsets)

def partition_by(inputs,attribute):
    groups=defaultdict(list)
    for input in inputs:
        key=input[0][attribute]
        groups[key].append(input)
    return groups

    
def partition_entropy_by(inputs,attribute):
    partitions=partition_by(inputs,attribute)
    return partition_entropy(partitions.values())

def classify(tree,input):
    if tree in [True,False]:
        return tree
    attribute,subtree_dict=tree
    subtree_key=input.get(attribute)
    if subtree_key not in subtree_dict:
        subtree_key=None
    subtree=subtree_dict[subtree_key]

    return classify(subtree, input)


def build_decision_tree(inputs,split_candidates=None):
    if split_candidates==None:
        split_candidates=inputs[0][0].keys()
    num_inputs=len(inputs)
    num_trues=len([label for _,label in inputs if label==True])
    num_falses=num_inputs-num_trues
    if not split_candidates:
        return num_trues >= num_falses
    best_attribute = min(split_candidates,
        key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates=[attribute for attribute in split_candidates if attribute != best_attribute]

    subtrees={attribute: build_decision_tree(subset,new_candidates) for attribute,subset in partitions.items()}
    subtrees[None] = num_trues > num_falses
    return (best_attribute, subtrees)


inputs = [
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'},
False),
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'},
False),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'},
True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'},
True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'},
True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'},
False),
({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'},
True),
({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'},
True),
({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'},
True),
({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'},
True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
]

attributes=inputs[0][0].keys()



if __name__=="__main__":
   tree = build_decision_tree(inputs)
   print(
    classify(tree, { "level" : "Junior",
    "lang" : "Java",
    "tweets" : "yes",
    "phd" : "no"} ))
