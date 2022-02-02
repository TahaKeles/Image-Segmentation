import math
import random
import numpy as np
import copy

# IMPORTANT: DON'T CHANGE OR REMOVE THIS LINE
#            SO THAT YOUR RESULTS CAN BE VISUALLY SIMILAR
#            TO ONES GIVEN IN HOMEWORK FILES
random.seed(5710414)

def euclidean(XA, XB):
  XA = np.array(XA)
  XB = np.array(XB)
  return np.sqrt(np.sum(np.square(XA - XB)))

def manhattan(XA, XB):
  XA = np.array(XA)
  XB = np.array(XB)
  return np.sum(np.abs(XA - XB))

class KMeans:
    def __init__(self, X, n_clusters, max_iterations=1000, epsilon=0.01, distance_metric="manhattan"):        
        self.X = X
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.clusters = []
        self.cluster_centers = []
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def choose_random_point(self, X):

      index = random.randrange(0,len(X))
      return X[index]

    def get_cluster_center(self,index):
      
      return self.cluster_centers[index]

    def random_init(self, X):

      initial_centroids = []
      for _ in range(self.n_clusters):
        rand_centroid = self.choose_random_point(X)
        initial_centroids.append(rand_centroid)
      return initial_centroids

    def fit(self):

      X = np.array(self.X)
      self.n_features = X[0].shape[0]
      self.cluster_centers = self.random_init(X)
      self.clusters = [[] for _ in range(self.n_clusters)]
      iteration = 0
      total_diff = float("inf")
      while iteration < self.max_iterations:
        print("KMeans iteration: ",iteration)
        current_cluster_members = [[] for _ in range(self.n_clusters)]
        for data_point in X:
          min_distance = float("inf")
          cluster = 0
          for cluster_idx, centroid_i in enumerate(self.cluster_centers):
              if self.distance_metric == "euclidean":
                distance = euclidean(centroid_i, data_point)
              else:
                distance = manhattan(centroid_i,data_point)
              if distance <= min_distance:
                  cluster = cluster_idx
                  min_distance = distance
          current_cluster_members[cluster].append(data_point)
        new_centroids = [[] for _ in range(self.n_clusters)]
        for cluster_i in range(self.n_clusters):
          new_centroid_i = np.zeros(self.n_features)
          members_of_current_cluster = current_cluster_members[cluster_i]
          if len(members_of_current_cluster) > 0:
            for member in current_cluster_members[cluster_i]:
              new_centroid_i = new_centroid_i + member
            new_centroid_i = new_centroid_i / len(members_of_current_cluster)
          else:
            new_centroid_i = self.choose_random_point(X)
          new_centroids[cluster_i] = new_centroid_i
        total_diff = float(0.0)
        for cluster_i in range(self.n_clusters):
          if self.distance_metric == "euclidean":
            total_diff = total_diff + euclidean(self.cluster_centers[cluster_i], new_centroids[cluster_i])
          else:
            total_diff = total_diff + manhattan(self.cluster_centers[cluster_i], new_centroids[cluster_i])


        self.cluster_centers = new_centroids
        self.clusters = current_cluster_members


        if total_diff <= self.epsilon:
          print("Epsilon boundary reached! Halting...")
          break
        iteration = iteration + 1
      

    def predict(self, instance):
      min_distance = float("inf")
      cluster = None
      for cluster_idx, centroid_i in enumerate(self.cluster_centers):
          distance = euclidean(centroid_i, instance)
          if distance <= min_distance:
              cluster = cluster_idx
              min_distance = distance
      return cluster