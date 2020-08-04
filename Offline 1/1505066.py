############# preprocess telco ###################

import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import log
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


def readTelcoFile():
    file = open('telco_dataset.csv')

    lines = file.readlines()

    count = 0
    cols = []
    dataset = []

    for line in lines:
        if count == 0:
            var = line.split(',')
            cols = [x.split('\n')[0] for x in var]
        else:
            var = line.split(',')
            var = [x.split('\n')[0] for x in var]
            if ' ' in var:
                continue
            
            c = 0
            for c in range(len(var)):
                if cols[c] == 'tenure':
                    var[c] = float(var[c])
                if cols[c] == 'MonthlyCharges':
                    var[c] = float(var[c])
                if cols[c] == 'TotalCharges':
                    if '.' in var[c]:
                        var[c] = float(var[c])
                    else:
                        var[c] = int(var[c])
                        var[c] = float(var[c])
            dataset.append(var)
        count += 1
    
    return dataset

def readAdultFile():
    name_file = open('adult.names')

    names = name_file.readlines()
    feature_names = []
    unique_feature_values = []

    for name in names:
        var = name.split(':')
        feature_names.append(var[0])
        var[1] = var[1].split(',')
        feature_values = [str(x).strip() for x in var[1]]
        unique_feature_values.append(feature_values)

    #print(unique_feature_values)

    name_file.close()


    # In[78]:


    name_file.close()

    file = open("adult.data")
    data_list = file.readlines()
    data_list = list(data_list)

    data_list.pop()

    numFeatures = len(feature_names)

    dataset = []

    for data in data_list:
        data = data.split(', ')

        if len(data) is not len(feature_names):
            print('Row Data Error')
            continue
        
        for i in range(numFeatures):
            if len(unique_feature_values[i])==1:
                data[i] = float(data[i])
            else:
                data[i] = data[i].strip('\n')
        
        if '?' in data:
            pass
        else:
            dataset.append(data)

    return dataset


def calculate_entropy(column):
    unique_features = set()
    for var in column:
        unique_features.add(var)
    
    count = []

    for feature in unique_features:
        c = 0
        for i in column:
            if i == feature:
                c += 1
        count.append(c)

    sum = 0
    total = len(column)
    for i in range(len(unique_features)):
        if (count[i]/total) == 0:
            pass
        elif (count[i]/total) == 1:
            pass
        else:
            sum -= (count[i]/total)*log((count[i]/total), 2)
        
    return sum


def calculate_information_gain(dataset,feature_name,class_name):
    set_entropy = calculate_entropy(dataset[class_name])

    total = len(dataset[feature_name])

    unique_features = set()
    for var in dataset[feature_name]:
        unique_features.add(var)

    count = []

    for feature in unique_features:
        c = 0
        for i in dataset[feature_name]:
            if i == feature:
                c += 1
        count.append(c)
    
    feature_wise_dataset = []
    
    for feature in unique_features:
        feature_wise_dataset.append(dataset.where(dataset[feature_name]==feature).dropna()[class_name])
    
    
    feature_entropy_sum = 0

    for i in range(len(count)):
        feature_entropy_sum += (count[i]/total)*calculate_entropy(feature_wise_dataset[i])
    
    ig = set_entropy - feature_entropy_sum 
    
    return ig


def telco_binarize_using_ig(dataset,cols):

    df = pd.DataFrame(dataset,columns=cols)

    for col in df.columns:
        if str(df[col].dtype) == 'float64':
            print(col)
            sorted_arr = np.sort(np.unique(df[col]))
            mid_point = []
            for i in range(len(sorted_arr)-1):
                mid_point.append((sorted_arr[i]+sorted_arr[i+1])/2)
            original = np.array(df[col])
            igs = []
            
            print(len(mid_point))
            
            count = 0
            for i in mid_point:
                print(i)
                arr = np.array(original).reshape(-1,1)
                arr = preprocessing.Binarizer(threshold=i).fit_transform(arr)
                df[col] = arr.reshape(1,-1)[0]
                igs.append(calculate_information_gain(df,col,df.columns[-1]))
                count += 1
            
            threshold = mid_point[np.argmax(igs)]
            arr = np.array(original).reshape(-1,1)
            arr = preprocessing.Binarizer(threshold=threshold).fit_transform(arr)
            df[col] = arr.reshape(1,-1)[0]

    df.to_csv('telco.csv')

def adult_binarize_using_ig(dataset,feature_names):
    df = pd.DataFrame(dataset,columns=feature_names)

    for col in df.columns:
        if str(df[col].dtype) == 'float64':
            sorted_arr = np.sort(np.unique(df[col]))
            mid_point = []
            for i in range(len(sorted_arr)-1):
                mid_point.append((sorted_arr[i]+sorted_arr[i+1])/2)
            original = np.array(df[col])
            igs = []
            count = 0
            for i in mid_point:
                print(count)
                arr = np.array(original).reshape(-1,1)
                arr = preprocessing.Binarizer(threshold=i).fit_transform(arr)
                df[col] = arr.reshape(1,-1)[0]
                igs.append(calculate_information_gain(df,col,df.columns[-1]))
                count += 1
            
            threshold = mid_point[np.argmax(igs)]
            arr = np.array(original).reshape(-1,1)
            arr = preprocessing.Binarizer(threshold=threshold).fit_transform(arr)
            df[col] = arr.reshape(1,-1)[0]

    df.to_csv('adult.csv')

def generate_custom_credit_card():
    df = pd.read_csv('creditcard.csv')
    df = df.drop(columns=df.columns[0:1])
    yes = df.where(df[df.columns[-1]]==1).dropna()
    yes = yes.reset_index()
    no_indexes = df.index[df[df.columns[-1]]==0].tolist() 
    values = df.values
    random_indexes = random.sample(no_indexes,20000)
    random_indexes = np.sort(random_indexes)
    new_dataset = []
    for i in random_indexes:
        new_dataset.append(values[i])
    new_dataset = pd.DataFrame(new_dataset,columns=df.columns)
    custom_dataset = new_dataset.append(yes)
    custom_dataset = custom_dataset.sample(frac=1)
    custom_dataset = custom_dataset.reset_index()
    custom_dataset = custom_dataset.drop(columns=custom_dataset.columns[-1])
    custom_dataset = custom_dataset.drop(columns=custom_dataset.columns[0])
    custom_dataset.to_csv('custom_credit_card.csv')

def credit_card_binarize_using_ig():
    df = pd.read_csv('custom_credit_card.csv')
    df = df.drop(columns=df.columns[0])
    np.unique(df[df.columns[-1]],return_counts=True)
    for col in df.columns:
        if str(df[col].dtype) == 'float64':
            sorted_arr = np.sort(np.unique(df[col]))
            mid_point = []
            for i in range(len(sorted_arr)-1):
                mid_point.append((sorted_arr[i]+sorted_arr[i+1])/2)
            original = np.array(df[col])
            igs = []
            
            total = len(mid_point)
            
            count = 0
            for i in mid_point:
                print(total-count)
                arr = np.array(original).reshape(-1,1)
                arr = preprocessing.Binarizer(threshold=i).fit_transform(arr)
                df[col] = arr.reshape(1,-1)[0]
                igs.append(calculate_information_gain(df,col,df.columns[-1]))
                count += 1
            
            threshold = mid_point[np.argmax(igs)]
            print(threshold)
            
            arr = np.array(original).reshape(-1,1)
            arr = preprocessing.Binarizer(threshold=threshold).fit_transform(arr)
            df[col] = arr.reshape(1,-1)[0]
            df.to_csv('final_custom_credit_card.csv')


def MaxclassValue(dataset,target_attribute_name):
    unique = set()
    
    for var in dataset[target_attribute_name]:
        unique.add(var)
    
    count = []
    
    unique = list(unique)
    
    for feature in unique:
        c = 0
        for var in dataset[target_attribute_name]:
            if var == feature:
                c += 1
        count.append(c)
    
    index = -1
    max = -1
    
    for i in range(len(count)):
        if count[i]>max:
            max = count[i]
            index = i
    
    return unique[index]  


############################ Decision Tree #########################################


def DecisionTree(current_dataset, feature_names, max_class_value, parent_node_majority=None):
    class_name = current_dataset.columns[-1]
    
    class_unique = set()
    for var in current_dataset[class_name]:
        class_unique.add(var)

    size = len(class_unique)
    class_unique = list(class_unique)

    if size == 1:
        return class_unique[0]
    elif len(current_dataset) == 0:
        return max_class_value
    elif len(feature_names) == 0:
        return parent_node_majority
    else:
        parent_node_majority = MaxclassValue(current_dataset, class_name)

        feature_wise_ig = []
        for feature in feature_names:
            feature_wise_ig.append(calculate_information_gain(
                current_dataset, feature, class_name))

        max = -1
        index = -1

        for i in range(len(feature_wise_ig)):
            if feature_wise_ig[i] > max:
                max = feature_wise_ig[i]
                index = i

        root = feature_names[index]
        new_features = []

        for i in feature_names:
            if i != root:
                new_features.append(i)
        unique = set()

        for var in current_dataset[root]:
            unique.add(var)
        unique = list(unique)

        tree = {}  # dictionary

        tree[root] = {}
        for each in unique:
            value_wise_dataset = current_dataset.where(
                current_dataset[root] == each).dropna()
            value_wise_tree = DecisionTree(
                value_wise_dataset, new_features, max_class_value, parent_node_majority)
            tree[root][each] = value_wise_tree

        return tree   


# In[31]:


def dfs(tree):
    if type(tree) == str:
        print(tree)
        print()
        return
    
    for node in tree.keys():
        print(node)
        dfs(tree[node])


# In[32]:


def predict_dt(tree,data,max_class):
    for node in list(data.keys()):
        if node in list(tree.keys()):
            try:
                result = tree[node][data[node]]
            except:
                return max_class
            result = tree[node][data[node]]
            if type(result) == dict:
                return predict_dt(result,data,max_class)
            else:
                return result


# In[33]:


dataset = pd.read_csv('telco.csv')
dataset = dataset.drop(columns=dataset.columns[0],axis=1)
dataset


# In[34]:


# Train Test Split

X = dataset.drop(columns=dataset.columns[-1])
Y = dataset[dataset.columns[-1]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

training_data = pd.concat([X_train,Y_train],axis=1)
testing_data = pd.concat([X_test,Y_test],axis=1)
'''
tree = DecisionTree(training_data,training_data.columns[:-1],max_target_value)

training_data = pd.read_csv('adult_train.csv')
training_data = training_data.drop(columns=training_data.columns[0],axis=1)

test_data = pd.read_csv('adult_test.csv')
test_data = test_data.drop(columns=test_data.columns[0],axis=1)
'''
max_class = MaxclassValue(training_data,training_data.columns[-1])

tree = DecisionTree(training_data,training_data.columns[:-1],max_class)


# In[35]:


print(tree)


# In[36]:


# Training Accuracy

arr = training_data[training_data.columns[-1]].values
predictions = []

data = training_data.iloc[:,:-1].to_dict(orient = "records")
count = 0
for d in data:
    prediction = predict(tree,d,max_class)
    predictions.append(prediction)
    count += 1

results = confusion_matrix(arr, predictions)

tn, fp, fn, tp = results.ravel()

print('Accuracy Score :',accuracy_score(arr, predictions)*100)

tpr = tp/(tp+fn)
print("True positive rate :",tpr*100)
tnr = tn/(tn+fp)
print("True negative rate :",tnr*100)
ppv = tp/(tp+fp)
print("Positive predictive value :",ppv*100)
fdr = fp/(tp+fp)
print("False discovery rate :",1-ppv*100)
f1 = 2*(ppv*tpr)/(ppv+tpr)
print("F1 score :",f1*100)


# In[39]:


# Testing Accuracy
arr = testing_data[testing_data.columns[-1]].values
predictions = []
data = testing_data.iloc[:,:-1].to_dict(orient = "records")

count = 0
for d in data:
    prediction = predict(tree,d,max_class)
    predictions.append(prediction)
    count += 1

results = confusion_matrix(arr, predictions)
    

tn, fp, fn, tp = results.ravel()

print('Accuracy Score :',accuracy_score(arr, predictions)*100)

tpr = tp/(tp+fn)
print("True positive rate :",tpr*100)
tnr = tn/(tn+fp)
print("True negative rate :",tnr*100)
ppv = tp/(tp+fp)
print("Positive predictive value :",ppv*100)
fdr = fp/(tp+fp)
print("False discovery rate :",1-ppv*100)
f1 = 2*(ppv*tpr)/(ppv+tpr)
print("F1 score :",f1*100)


# In[ ]:









############################# Adaboost ##############################################


def DecisionStump(current_dataset, feature_names):
    class_name = current_dataset.columns[-1]
    class_unique = set()
    for var in current_dataset[class_name]:
        class_unique.add(var)

    size = len(class_unique)
    class_unique = list(class_unique)
    
    if size == 1:
        return class_unique[0]
    elif len(feature_names) == 0:
        return MaxclassValue(current_dataset, class_name)
    else:
        feature_wise_ig = []
        for feature in feature_names:
            feature_wise_ig.append(calculate_information_gain(
                current_dataset, feature, class_name))

        max = -1
        index = -1

        for i in range(len(feature_wise_ig)):
            if feature_wise_ig[i] > max:
                max = feature_wise_ig[i]
                index = i

        root = feature_names[index]
        new_features = []

        for i in feature_names:
            if i != root:
                new_features.append(i)
        unique = set()

        for var in current_dataset[root]:
            unique.add(var)
        unique = list(unique)

        tree = {}  # dictionary

        tree[root] = {}
        for each in unique:
            value_wise_dataset = current_dataset.where(
                current_dataset[root] == each).dropna()
            if(len(value_wise_dataset)>0):
                value_wise_tree = DecisionStump(
                value_wise_dataset,[])
                tree[root][each] = value_wise_tree

        return tree


# In[58]:


def predict(tree,data):
    if type(tree) == str:
        return tree
    
    for key in list(data.keys()):
        if key in list(tree.keys()):
            result = None
            if data[key] in tree[key].keys():
                result = tree[key][data[key]]
            
            if type(result) == dict:
                return predict(result,data)
            else:
                return result


# In[59]:


def resample(dataset,sample_weight):
    indexed_dataset = dataset.values
    indexes = np.random.choice(np.arange(0,len(indexed_dataset)),len(indexed_dataset),p=sample_weight)
    
    new_dataset = []
    for i in indexes:
        new_dataset.append(indexed_dataset[i])
    
    new_dataset = pd.DataFrame(new_dataset,columns=dataset.columns)
    return new_dataset


# In[60]:


def Adaboost(dataset,K):
    arr_stump = []
    z = []
    sample_weight = [1/len(dataset) for i in range(len(dataset))]
    length = len(dataset)
    data = dataset
    arr = training_data[training_data.columns[-1]].values
    
    for k in range(K):
        data = resample(dataset,sample_weight)
        ds = DecisionStump(data,data.columns[:-1])
        print(ds)
        data = dataset.iloc[:,:-1].to_dict(orient = "records")
        count = 0
        accuracy = 0
        error = 0
        predictions = []
        
        
        for d in data:
            prediction = predict(ds,d)
            
            predictions.append(prediction)
            
            if prediction == arr[count]:
                accuracy += 1
            else:
                error += sample_weight[count]
            count += 1
        if error > .5:
            continue
        
        arr_stump.append(ds)
        for i in range(len(dataset)):
            if arr[i] == predictions[i]:
                sample_weight[i] = sample_weight[i]*(error/(1-error))
        
        sample_weight = [ x/np.sum(sample_weight) for x in sample_weight]
        z.append(log(((1-error)/error),2))
        
    return arr_stump,z


# In[61]:


dataset = pd.read_csv('final_custom_credit_card.csv')
dataset = dataset.drop(columns=dataset.columns[0],axis=1)
dataset


# In[62]:


# Train Test Split
X = dataset.drop(columns=dataset.columns[-1])
Y = dataset[dataset.columns[-1]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 10)

training_data = pd.concat([X_train,Y_train],axis=1)
testing_data = pd.concat([X_test,Y_test],axis=1)
'''

training_data = pd.read_csv('adult_train.csv')
training_data = training_data.drop(columns=training_data.columns[0],axis=1)

testing_data = pd.read_csv('adult_test.csv')
testing_data = test_data.drop(columns=test_data.columns[0],axis=1)
'''


# In[ ]:


max_class = MaxclassValue(training_data,training_data.columns[-1])
ds,z = Adaboost(training_data,20)


# In[ ]:


#training data accuracy

unique_class = np.unique(training_data[training_data.columns[-1]])
data = training_data.iloc[:,:-1].to_dict(orient = "records")

arr = training_data[training_data.columns[-1]].values
predictions = []

count = 0

for d in data:
    result = np.zeros(len(unique_class))
    for k in range(len(ds)):
        pred = predict(ds[k],d)
        for i in range(len(unique_class)):
            if pred==unique_class[i]:
                result[i] += z[k]
    prediction = unique_class[np.argmax(result)]
    predictions.append(prediction)
    
    count += 1

results = confusion_matrix(arr, predictions)

tn, fp, fn, tp = results.ravel()

print('Accuracy Score :',accuracy_score(arr, predictions)*100)

tpr = tp/(tp+fn)
print("True positive rate :",tpr*100)
tnr = tn/(tn+fp)
print("True negative rate :",tnr*100)
ppv = tp/(tp+fp)
print("Positive predictive value :",ppv*100)
fdr = fp/(tp+fp)
print("False discovery rate :",tnr*100)
f1 = 2*(ppv*tpr)/(ppv+tpr)
print("F1 score :",f1*100)


# In[ ]:


#testing data accuracy

unique_class = np.unique(testing_data[testing_data.columns[-1]])
data = testing_data.iloc[:,:-1].to_dict(orient = "records")

arr = testing_data[testing_data.columns[-1]].values
predictions = []

count = 0

for d in data:
    result = np.zeros(len(unique_class))
    for k in range(len(ds)):
        pred = predict(ds[k],d)
        for i in range(len(unique_class)):
            if pred==unique_class[i]:
                result[i] += z[k]
    prediction = unique_class[np.argmax(result)]
    predictions.append(prediction)
    
    count += 1 

results = confusion_matrix(arr, predictions)

#tn, fp, fn, tp = results.ravel()

print('Accuracy Score :',accuracy_score(arr, predictions)*100)

tpr = tp/(tp+fn)
print("True positive rate :",tpr*100)
tnr = tn/(tn+fp)
print("True negative rate :",tnr*100)
ppv = tp/(tp+fp)
print("Positive predictive value :",ppv*100)
fdr = fp/(tp+fp)
print("False discovery rate :",tnr*100)
f1 = 2*(ppv*tpr)/(ppv+tpr)
print("F1 score :",f1*100)


# In[ ]: