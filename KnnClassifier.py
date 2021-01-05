# This is an implementation of KNN(k-nearest neighbor) algorithm from scratch

import numpy as np

class KnnClassifier:
    
    # defining constructor 
    def __init__(self,n_neighbors= 5):
        self.n_neighbors = n_neighbors
        
    # This function is used for training the model, also called as fitting 
    # In KNN algorithm all the magic happens when we ask the model for predictions
    def fit(self, X,y):
        self._X = (X - X.mean()) / X.std() # standardisation
        self._y = y
        
      
    def point_classification(self, point):
        """
        Here it tells is to which class/group the given point belongs to..  
        The procedure for finding K nearest Neighbour is as follows:
        1. Calculate the distance of the point from the rest 
        2. Sort the acquired data obtained according to the distance
        3. Keep only the initial "n_neighbors" from the above sorted data
            (in simple terms "n_neighbors" means, how many comparisions a point should make 
            from the neareast points in order to take a decission!)
        4. Atlast the most common class/label/group among those K entries will be the class of the data point.
        
        """
        
        distance_class = [] # In this list we store the: [[distance from points, it's group]]
        
        for x, y in zip(self._X, self._y):
            distance = np.sqrt((point - np.array(x)) ** 2).sum()
            distance_class.append([distance,y])
        
        # sorting the list according to the distances 
        sorted_distance_class = sorted(distance_class)
        # Now initial specified "n_neighbors" are sliced 
        required_k_nearest = sorted_distance_class[:self.n_neighbors] 
        
        # returning the nearest class/label/group as per the majority count!
        groups, counts = np.unique(np.array(required_k_nearest)[:, 1], return_counts=True)
        nearest_group = groups[np.argmax(counts)]
        return nearest_group
        
    
    # Give the numpy array prediction of points passed
    def predict(self, X):
        predictions = []
        X = (X - X.mean()) / X.std() # standardisation
        for point in X:
            predictions.append(self.point_classification(point))
            
        return np.array(predictions)
    