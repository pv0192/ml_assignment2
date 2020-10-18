import pandas 
import numpy 
from pprint import pprint  

def ID3(df,origina_df,features,target,parent_node_class = None):  
    #Setting default parent node class as None
    #Define end scnarios --> If true, return leaf
      
    #If all target_values have the same value, return this value  
    if len(numpy.unique(df[target])) <= 1:  
        return numpy.unique(df[target])[0]  
      
    #If empty, return the  target feature value in the original set  
    elif df is None: 
        target_feature_value =  numpy.argmax(numpy.unique(original_df[target],return_counts=True)[1])
        return numpy.unique(original_df[target])[target_feature_value]  
      
    #If the feature space is empty, return the target feature value of the direct parent node
      
    elif len(features) is 0:  
        return parent_node_class  
      
    #If none of the above is true, tree grows!  
      
    else:  
        #Set the default value for this node 
        feature_unuique_values = numpy.unique(df[target])
        feature_unique_values_count = numpy.unique(df[target],return_counts=True)[1]
        parent_node_class = feature_unuique_values[numpy.argmax(feature_unique_values_count)]  
        
        #Select the feature which top splits the df  
        #find infogain  for the features in the df 
        feature_values = []
        for feature in features:
            feature_values = information_gain(df,feature,target)    
        top_feature = features[numpy.argmax(feature_values)]  
          
        #Create the tree structure. 
        tree = {top_feature:{}}  
          
          
        #Remove the feature with the top inforamtion gain from the feature space
        
        features = [i for i in features if i != top_feature]  
          
        #Grow a branch under the root node for each possible value of the root node feature  
          
        for value in numpy.unique(df[top_feature]):  
            value = value  
            #Split the df along the value of the feature with the largest information gain and therwith create sub_dfs  
            sub_df = df.where(df[top_feature] == value).dropna()  
              
            #Call the ID3 algorithm for each of those sub_dfs with the new parameters  
            sub_tree = ID3(sub_df,df,features,target,parent_node_class)  
              
            #Add the sub tree, grown from the sub_df to the tree under the root node  
            tree[top_feature][value] = sub_tree  
              
        return(tree)

def calculate_entropy(col):  
    
    
    attributes,counts = numpy.unique(col,return_counts = True)

    entropy = 0
    for i in range(len(attributes)):
        entropy += (-counts[i]/numpy.sum(counts))*numpy.log2(counts[i]/numpy.sum(counts))

    return entropy  
  
def information_gain(df,feature_name_split,target_name):  
         
    #entropy of total df
    entropy_total = calculate_entropy(df[target_name])  
      
    ##Calculate entropy of the df  
      
    #Calculate vals,counts for splits  
    vals,counts= numpy.unique(df[feature_name_split],return_counts=True)  
      
    #Calculate the weighted entropy

    entropy_weighted = 0
    for i in range(len(vals)):
        probability = counts[i]/numpy.sum(counts)
        scenario_entropy = calculate_entropy(df.where(df[feature_name_split]==vals[i]).dropna()[target_name])
        entropy_weighted += (probability*scenario_entropy)


    #info gain
    Info_Gain = entropy_total - entropy_weighted 
    return Info_Gain 

                  
def predict(query,tree,default = 1):  
    
    for i in list(query.keys()):  
        if i in list(tree.keys()):  
            
            try:  
                result = tree[i][query[i]]   
            except:  
                return default  
    
            result = tree[i][query[i]]  
            
            if isinstance(result,dict):  
                return predict(query,result)  
            else:  
                return result  
  
def df_split(df):  
    #redo indexes with reset_index
    training = df.iloc[:85]
    testing = df.iloc[85:]  
    return training,testing  
     
  
def test(df,tree):  
    #convert df to dict  
    query = df.iloc[:,:-1].to_dict(orient = "records")  
      
    #store predictions  
    predicted = pandas.DataFrame(columns=["predict"])   
      
    #predict + calculate prediction % accuracy  
    for i in range(len(df)):
        predicted.loc[i,"predict"] = int(predict(query[i],tree,1.0))
        print("Row" + str(i) + " class: " + str(df.loc[i,"class"]))
        print("Row"+ str(i) + " prediction: " + str(predicted.loc[i,"predict"]))

    #save predictions
    final_predictions = predicted["predict"]
    df.join(final_predictions).to_csv('predictions.csv')


    correct_predictions = (numpy.sum(predicted["predict"] == df["class"])/len(df))*100

    print("Total rows: " , len(df) )
    print("Total predicted correctly: ", numpy.sum(predicted["predict"] == df["class"]))
    print('The prediction accuracy is: ', correct_predictions,   '%' ) 
      

def main():
    #import the df 
    df = pandas.read_csv('animal_classifier.csv',  
                      names=['animal_name','hair','feathers','eggs','milk',  
                                                   'airbone','aquatic','predator','toothed','backbone',  
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','class',]) 

    #drop name as it is not a classifier  
    df=df.drop('animal_name',axis=1)  
    data_split = df_split(df)
    training_df = data_split[0].reset_index(drop=True)
    testing_df = data_split[1].reset_index(drop=True)
    tree = ID3(training_df,training_df,training_df.columns.values.tolist(),"class")  
    pprint(tree)  
    test(testing_df,tree)


if __name__ == "__main__":
    main() 