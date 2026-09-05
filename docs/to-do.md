Okay, now write the list_processor.py. where you can see i have the list of steps, plus their class methods.  . 



what you need to do is, write the code down. simple, that takes the data, loads in dataframes, first one is the real data, apply all the outlier removals one by one separately, meaning that once first method, save that, and then second method, on old dataframe, and we get two dataframes



then apply the imputers, one by one, on all dataframes from before



so, example, there are 2 outliers, and 2 imputers, we get

2*2 = 4 dataframes



all of this is happening on the dataframe taken from csv1



then apply the smoothers. keep the name at each step



then do the train val test split, and do this by the subject_id in the dataframe (of csv1)




Do PCA for the features. check it. and then do PCA. what does that give you?


Do check that the eval metrics are working
and do the dimensionality checks. only vitals, only age sex things. and then apply the models. and then check how do these work??


add the save_csv_file feature at points that are needed. 


add the functionality that you can train and test everything on set-features only..