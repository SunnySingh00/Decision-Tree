import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        max_ig = float('-inf')
        local_feature = []
        ig_array = []
        for i in range(0,len(self.features[0])):
            branches = self.getbranches((np.unique(np.array(self.features)[:,i])),i,self.features,self.labels,(np.unique(np.array(self.labels))))
            parent = self.getparententropy(self.labels)
            ig = Util.Information_Gain(parent,branches)
            ig_array.append(ig)
            if max_ig<ig: # entropy is greater
                self.dim_split = i
                max_ig = ig
                local_feature = (np.unique(np.array(self.features)[:,i])).tolist()
            elif max_ig==ig: # entropy is same
                if len((np.unique(np.array(self.features)[:,i])).tolist())>len(local_feature): # select features with greater attributes
                    self.dim_split = i
                    local_feature = (np.unique(np.array(self.features)[:,self.dim_split])).tolist()
                elif len((np.unique(np.array(self.features)[:,i])).tolist()) == len(local_feature): # same attributes
                    if self.dim_split>=i: # select with lower index
                        self.dim_split=i
                        local_feature = (np.unique(np.array(self.features)[:,self.dim_split])).tolist()   
                        
        
        if max(ig_array)==0.0:
            self.splittable=False
            return
        else:
            self.feature_uniq_split = local_feature
            feature_selected = np.array(self.features)[:, self.dim_split]
            modf_feature = np.delete(np.array(self.features),self.dim_split,1)
            for fs in np.sort(self.feature_uniq_split):
                indexes = np.where(fs == feature_selected)
                x_fs = modf_feature[indexes].tolist()
                l_fs = np.array(self.labels)[indexes].tolist()
                child = TreeNode(x_fs,l_fs,self.num_cls)
                self.children.append(child)
                if len(x_fs) == 0 or len(x_fs[0])==0:
                     child.splittable = False
        for child in self.children:
            if child.splittable:
                child.split()
        return

        raise NotImplementedError
    
    def getparententropy(self,labels):
        uniq_labels = np.unique(labels)
        res = {}
        for i in labels:
            if i in res:
                res[i]+=1
            else:
                res[i]=1
        tot=0
        for l,val in res.items():
            tot+=val
        ent=0
        for l,val in res.items():
            ent+=-(val/tot)*(np.log2(val/tot))

        return ent
    
    
    def getbranches(self,feature_uniq_split,index,features,labels,labels_unique):
        branches = []
        for fs in feature_uniq_split:
            arr = [0 for i in range(0,len(labels_unique))]
            for i in range(0,len(features)):
                if fs==features[i][index]:
                    arr[np.where(labels[i]==labels_unique)[0][0]]+=1
            branches.append(arr)
        return branches
    def getconditionalentropy(self,branches):
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
        return avg

    # TODO: predict the branch or the class
    def predict(self,feature):
        # feature: List[any]
        # return: int
        if self.splittable:

            modf_feature = np.delete(np.array(feature),self.dim_split)
            index = self.feature_uniq_split.index(feature[self.dim_split])

            return self.children[index].predict(modf_feature)
        else:
            #print(self.cls_max)
            return(self.cls_max)
        raise NotImplementedError