import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    avg = 0
    list_tot = 0
    for branch in branches:
        list_tot +=sum(branch)
    for branch in branches:
        tot = sum(branch)
        if tot!=0:
            ent=0
            for child in branch:
                if child!=0:
                    ent+= -(child/tot)*(np.log2(child/tot))
                else:
                    ent+=0
            avg+=(tot/list_tot)*(ent)
    #print(S-avg)
    return S-avg


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    ypred = decisionTree.predict(X_test)
    accuracy = get_accuracy(y_test, ypred)
    stack = traverse(decisionTree.root_node, [decisionTree.root_node])[::-1]
    print(stack)
    for node in stack:
            node.splittable = False
            curr_accuracy = get_accuracy(decisionTree.predict(X_test), y_test)
            if  curr_accuracy > accuracy:
                accuracy = curr_accuracy
            elif curr_accuracy == accuracy:
                if node!=decisionTree.root_node:
                    accuracy = curr_accuracy
            else:
                node.splittable = True

def traverse(node,array):
    if node.splittable:
        for child in node.children:
            array.append(child)
            traverse(child, array)
    return array

def get_accuracy(ytest,ypred):
    yes = 0
    for i in range(len(ytest)):
        if ytest[i] == ypred[i]:
            yes += 1
    return float(yes/len(ytest))

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
