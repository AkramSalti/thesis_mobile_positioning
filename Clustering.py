import numpy as np
import time
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.cluster.hierarchy import fclusterdata
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
import DataProcessor
from Grid_backend import Grid, calculateEstimatedPositions
from scipy.spatial.distance import cdist
from Unified_functions import mydist
from pyclustering.cluster.kmedoids import kmedoids


"reading data into dataframes"
measurements_denseurban=DataProcessor.getDataFromCSV(r"LTE_denseurban.tsv")
measurements_suburban=DataProcessor.getDataFromCSV(r"LTE_suburban.tsv")

df_denseurban=DataProcessor.getDataFrameBasedOnParameters(measurements_denseurban,['Lat','Long','CID-S','CID-N1','CID-N2','CID-N3','CID-N4','CID-N5','CID-N6','RSRP-S','RSRP-N1','RSRP-N2','RSRP-N3','RSRP-N4','RSRP-N5','RSRP-N6'])
df_suburban=DataProcessor.getDataFrameBasedOnParameters(measurements_suburban,['Lat','Long','CID-S','CID-N1','CID-N2','CID-N3','CID-N4','CID-N5','CID-N6','RSRP-S','RSRP-N1','RSRP-N2','RSRP-N3','RSRP-N4','RSRP-N5','RSRP-N6'])

df_denseurban_measurements=df_denseurban[:100]
df_suburban_measurements=df_suburban[:100]


df_measurements=DataProcessor.convertDataFrameForPositioning(df_denseurban_measurements)


df_calibration, df_test = train_test_split(df_measurements, test_size=0.2)
df_calibration = DataProcessor.reindexDataFrame(df_calibration)
df_test = DataProcessor.reindexDataFrame(df_test)


df_clustering=DataFrame(df_calibration,columns=['Lat','Long'])

df_calibration['Lat']*= 10 ** 6
df_calibration['Long']*= 10 ** 6
df_test['Lat']*= 10 ** 6
df_test['Long']*= 10 ** 6
df_calibration = df_calibration.astype({'Lat': 'int64', 'Long': 'int64'})
df_test = df_test.astype({'Lat': 'int64', 'Long': 'int64'})
df_calculations=DataFrame(df_calibration,columns=['Lat','Long'])

"kmean clustering"
cluster_labels=[]
startKmeans=time.time()
kmeans = KMeans(n_clusters=8,init='k-means++').fit(df_clustering)
endKmeans=time.time()
execTimeKmeans=endKmeans-startKmeans
print("Kmean execution time: "+str(round(execTimeKmeans*1000,3))+"ms")
for i in kmeans.labels_:
    cluster_labels.append(i)

"kmedoids clustering"
"""
listPoint=df_clustering.values.tolist()
kmedoids = kmedoids(data=listPoint,initial_index_medoids=[0,20,35,60,85])
kmedoids.process()
clusters=kmedoids.get_clusters()
cluster_labels = np.zeros((len(listPoint),),dtype=int)
for i in range(len(clusters)):
  for x in np.nditer(np.asarray(clusters[i])):
     cluster_labels[x] = i
"""
"""
"fclusterdata clustering"
"methods 'single' requires that the distance between nearest two points should be at most equal t in the same cluster"
"method 'complete' requires that the distance between furthest two points in a cluster should be at most equal t"
"criterion determines whether distance based 'distance' or cluster number based 'maxclust' clustering is needed"
fcluster_clustering = fclusterdata(df_clustering,t=15, metric=mydist,criterion='distance',method='complete')
cluster_labels=[]
for i in range(len(fcluster_clustering)):
    label=fcluster_clustering[i]-1
    cluster_labels.append(label)
"""

"""
"Agglomerative clustering based on cluster number"
agg_clustering= AgglomerativeClustering(n_clusters=5,linkage='complete',distance_threshold=None).fit(df_clustering)

"Agglomerative clustering based on distance. distance matrix needed to be passed instead of dataframe. Method is same as in fclusterdata"
c=cdist(df_clustering,df_clustering,metric=mydist)
rfunc=np.vectorize(round)
dmatrix=rfunc(np.reshape(c,(len(df_clustering),len(df_clustering))),1)
agg_clustering= AgglomerativeClustering(n_clusters=None,linkage='single',affinity='precomputed',compute_full_tree=True,distance_threshold=2).fit(dmatrix)

labels=agg_clustering.labels_
cluster_labels=[]
for i in labels:
    cluster_labels.append(i)
"""

"create list of dataframes. Each dataframe contains a cluster and its measurements"
df_calculations['Cluster']=cluster_labels
df_cluster_list=[]
labelsSet=list(dict.fromkeys(cluster_labels))
for i in range(len(cluster_labels)):
    df_single_cluster=DataFrame(columns=['Lat','Long','Cluster'])
    for j in range(len(cluster_labels)):
        if (cluster_labels[i] == cluster_labels[j]):
            df_single_cluster = df_single_cluster.append({'Lat': df_calculations['Lat'].get(j), 'Long': df_calculations['Long'].get(j), 'Cluster': df_calculations['Cluster'].get(j)}, ignore_index=True)
            df_single_cluster = df_single_cluster.astype({'Cluster': 'int'})
    if (cluster_labels[i] in labelsSet):
        df_cluster_list.append(df_single_cluster)
    if(cluster_labels[i] in labelsSet):
        labelsSet.remove(cluster_labels[i])


"calculate min and max distance inside each cluster"
"calculating distance matrix takes a long time, better to turn it off when not needed"
for i in range(len(df_cluster_list)):
    df_cluster= DataFrame(df_cluster_list[i],columns=['Lat','Long'])
    df_cluster['Lat']/=10**6
    df_cluster['Long']/=10**6
    cluster_Dis_matrix = cdist(df_cluster, df_cluster, metric=mydist)
    dist_list=[]
    max_dist_list=[]
    min_dist_list=[]
    max_dist, min_dist = 0.0, 0.0
    for a in range(len(cluster_Dis_matrix)):
        for b in range(len(cluster_Dis_matrix)):
            if (cluster_Dis_matrix[a][b] != 0):
                dist_list.append(cluster_Dis_matrix[a][b])
    if (len(dist_list) != 0):
        max_dist = max(dist_list)
        min_dist = min(dist_list)
    for j in range(len(df_cluster)):
        max_dist_list.append(max_dist)
        min_dist_list.append(min_dist)
    df_cluster_list[i]['Max_Dist']=max_dist_list
    df_cluster_list[i]['Min_Dist']=min_dist_list


"calculate center points of each cluster and add it to GridCenters list to be used at posiotioning"
labels_set = np.unique(cluster_labels)
centers = np.ones((len(labels_set), 2), dtype='int')
for i in range(len(df_cluster_list)):
    clusterID = df_cluster_list[i]['Cluster'].get(0)
    centers[clusterID] = df_cluster_list[i][['Lat', 'Long']].mean(axis=0)
GridCenters = centers[cluster_labels]

"adding centers to clusters list"
for i in range(len(df_cluster_list)):
    center_lat = []
    center_long = []
    for j in range(len(df_cluster_list[i])):
        clusterID = df_cluster_list[i]['Cluster'].get(0)
        center_lat.append(centers[clusterID][0])
        center_long.append(centers[clusterID][1])
    df_cluster_list[i]['Center_Lat'] = center_lat
    df_cluster_list[i]['Center_Long'] = center_long
endAddingCenters=time.time()

"creating a dataframe that describe all the clusters df_stats"
clusterSize=np.bincount(cluster_labels)
avg_lat_set=[]
avg_long_set=[]
cluster_ID=[]
max_dist=[]
min_dist=[]
cluster_size=[]
df_stats=DataFrame()
for i in range(len(df_cluster_list)):
    df_clus=DataFrame(df_cluster_list[i])
    for j in range(len(df_clus)):
        if(df_clus['Cluster'].get(j) not in cluster_ID):
            cluster_ID.append(df_clus['Cluster'].get(j))
            avg_lat_set.append(df_clus['Center_Lat'].get(j))
            avg_long_set.append(df_clus['Center_Long'].get(j))
            max_dist.append(df_clus['Max_Dist'].get(j))
            min_dist.append(df_clus['Min_Dist'].get(j))
            cluster_size.append(len(df_clus))
df_stats['ClusterID']=cluster_ID
df_stats['Size']=cluster_size
df_stats['Center_Lat']=avg_lat_set
df_stats['Center_Long']=avg_long_set
df_stats['Max_Dist']=max_dist
df_stats['Min_Dist']=min_dist
endStats=time.time()

"Grid the data based on the calculated clusters, and compute positions for test measurements"
gd = Grid(df_calibration,GridCellCenters=GridCenters)
gd.fitGaussianDistributions(min_variance=3.0)
gd.calculateCellSelection(most_common=2)
start = time.perf_counter()
dfr = gd.localization_DataFrame(df_test, max_workers=1)
dfr = calculateEstimatedPositions(dfr, [3,6,9,20])
stop = time.perf_counter()
print(dfr.filter(regex='Error_w').describe().to_string())

gdNoCenters = Grid(df_calibration)
gdNoCenters.fitGaussianDistributions(min_variance=3.0)
gdNoCenters.calculateCellSelection(most_common=2)
start = time.perf_counter()
dfrNoCenters= gdNoCenters.localization_DataFrame(df_test, max_workers=1)
dfrNoCenters = calculateEstimatedPositions(dfrNoCenters, [3,6,9,20])
stop = time.perf_counter()
print(dfrNoCenters .filter(regex='Error_w').describe().to_string())


pointNoCenters=list(gdNoCenters.grid.keys())
latlist=[]
longlist=[]
for i in pointNoCenters:
    lat=i[0]/(10**6)
    long=i[1]/(10**6)
    latlist.append(lat)
    longlist.append(long)
dfpointsNocenters=DataFrame()
dfpointsNocenters['Lat']=latlist
dfpointsNocenters['Long']=longlist

pointCenters=list(gd.grid.keys())
latlist=[]
longlist=[]
for i in pointCenters:
    lat=i[0]/(10**6)
    long=i[1]/(10**6)
    latlist.append(lat)
    longlist.append(long)
dfpointscenters=DataFrame()
dfpointscenters['Lat']=latlist
dfpointscenters['Long']=longlist
points_centers_length=len(dfpointscenters)

dfplotTest=DataFrame()
dfplotTest['Lat']=df_test['Lat']
dfplotTest['Long']=df_test['Long']
dfplotTest['Lat']/=10**6
dfplotTest['Long']/=10**6

df_pointscenters=dfpointscenters.append(dfpointsNocenters,ignore_index=True)
df_pointscenters=df_pointscenters.append(dfplotTest,ignore_index=True)

plot_list=[0.0 for i in range(0,len(df_pointscenters))]
for i in range(len(plot_list)):
    if(i>=points_centers_length and i<(points_centers_length+len(dfpointsNocenters))):
        plot_list[i]=1.0
    if(i>=(points_centers_length+len(dfpointsNocenters))):
        plot_list[i]=2.0

"plotting the grid cells with test measurements on denseurban map"
BBox=(130.4129,130.4205,33.58991,33.5946)
mapImage = plt.imread('map1000.png')
fig, ax = plt.subplots(figsize = (7,7))
ax.scatter(df_pointscenters['Long'], df_pointscenters['Lat'], c=plot_list, s=50, alpha=0.9)
#ax.scatter(df_clustering['Long'], df_clustering['Lat'], c=cluster_labels, s=50, alpha=0.9)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(mapImage, zorder=0 , extent = BBox, aspect='equal')
plt.show()

