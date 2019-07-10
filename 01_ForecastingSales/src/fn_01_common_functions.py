#----------------------------------------------------------------------------------------
# FUNCTIONS
#----------------------------------------------------------------------------------------

# 01 function
#----------------------------------------------------------------------------------------
def cartesian_lists_toDF (list1, list2, list3, list4, col_names):
    """
    Calculate the cartesian join of 3 lists and return a dataframe

    list1: first list
    list2: second list
    list3: third list
    list4: fourth list
    col_names: column names when converting to dataframe
    """
    # creating the cartesian product of lists
    cartesian_list = [[a,b,c,d] for a, b, c, d in product(list1, list2, list3, list4)]
    
    # return the dataframe
    return pd.DataFrame(cartesian_list, columns=col_names)

# 02 function
#----------------------------------------------------------------------------------------
def calculate_correlations(input_data, index_col, cat_features, exclu_elements):
    """
    Calculate the correlation matrix

    input_data: dataframe to be used for the correlation matrix (keep only the columns 
           used in the correlations)
    index_col: column which will be transposed to find correlation
    cat_features: features to do OnHotEncoding
    exclu_elements: elements to be excluded from the final correlated items.
                    For any reason, these elements shouldn't be present in the 
                    correlated items

    """    
    try:
      # encode the categorical features
      encoded_data = pd.get_dummies(input_data,columns=cat_features,drop_first=True)

      pd_transposed_data = encoded_data.set_index('Style_display_code').T

      # get the number of items
      items_list = [str(a) for a in pd_transposed_data.columns]

      print("Number of items to correlate :{}_Timestamp:{}".format(str(len(items_list)), 
                                                              format(str(datetime.now()))))
        

      #compute correlations and save the pickle file
#      matrix = pd_transposed_data.corr().values
#      pickle.dump(matrix, open(staging_dir+ '/corr_matrix_output_py3.p', 'wb'))
      
      # read from the saved pickle file - ONLY FOR CONSECUTIVE RUNS, TO SAVE TIME
      matrix = pickle.load(open(staging_dir+ '/corr_matrix_output_py3.p', "rb" ) )

      print("Corr Matrix size:{}_Timestamp:{}".format(str(matrix.size),
                                                      format(str(datetime.now()))))

    except Exception as e:
      print(" Error !!", e)
    
    # return the top correlated items
    return top_correlateditems(items_list,matrix, index_col, exclu_elements)

# 03 function
#----------------------------------------------------------------------------------------
def top_correlateditems(elements_list,corr_matrix,col_name,exclu_elements,n_upper=10):
    """
    input

    corr_object: contain the list of elements in the correlation matrix
    correlations: matrix output from corr() function
    col_name: column name of which correlation is found- used in final dataframe column
    n_upper:  count of top correlated elements. Default is 10

    output
    pandas dataframe containing for each items the correlated items

    """  
    values=[]
    keys=[]

    #loop through the elements
    for element in elements_list:
        
#        # DEBUG log
#        print("Item correlated :{}_Timestamp:{}".format(str(element), 
#                                                            format(str(datetime.now()))))

        #gets the index (in the correlation matrix for that element)
        i=elements_list.index(element)

        #create 1d array for that product
        oned=corr_matrix[i,:]

        #select the top correlation (indexes)
        topn_indices=np.flip(np.argsort(oned)[-n_upper-1:],axis=0)


        #select the bottom correlation (indexes)
        #bottomn_indices=np.argsort(corr_matrix[i,:])[:n_lower]

        #indices for this run
        #indices=np.concatenate((topn_indices,bottomn_indices),axis=0)

        #values of the indices of the related columns 
        values.append([elements_list[a] for a in topn_indices 
                       if elements_list[a] not in exclu_elements])        
        #while element in values[i]: values[i].remove(element)
          
        

        #keys
        keys.append(element)

    #produces a dictionary: given a certain element which other elements are correlated
    my_dict=dict(zip(keys,values))
    
    #convert into panda df
    pd_df=pd.DataFrame([(a,b,str(my_dict[a].index(b)))
                        for a in my_dict.keys() for b in my_dict[a]],
                       columns=[col_name,"correlated_"+col_name,"order"])

    #convert into spark
    #association=spark.createDataFrame(pd_df)
    
    print(" Finished calculating correlations_Timestamp:{}" \
          .format(format(str(datetime.now()))))

    return pd_df

# 04 function
#----------------------------------------------------------------------------------------
def anti_join(x, y, on):
    """Return rows in x which are not present in y"""
    ans = pd.merge(left=x, right=y, how='left', indicator=True, on=on)
    ans = ans.loc[ans._merge == 'left_only', :].drop(columns='_merge')
    return ans

# 05 function
#----------------------------------------------------------------------------------------
def anti_join_all_cols(x, y):
    """Return rows in x which are not present in y"""
    assert set(x.columns.values) == set(y.columns.values)
    return anti_join(x, y, x.columns.tolist())

