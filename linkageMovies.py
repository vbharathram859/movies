# K means had noticeably better results than centroid based linkage

import math
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
import pickle

def readFile(fname):
    global cols
    df = pd.read_csv(fname)
    cols = list(df.columns.values)
    del df[cols[0]]
    cols = cols[1:]
    return df

def genClosestDict(clusters, using):
    curCenters = [clusters[i][1] for i in using]
    tree = spatial.KDTree(curCenters)
    closestDict = {}

    for i in using:  # go through all possible pairs of clusters to find the two closest to each other
        print(i)
        curVal,curInd = tree.query(clusters[i][1], 2) # calculate square distance, take two points because the closest point is always itself
        if curInd[0] == i:  #
            curVal = curVal[1]
            curInd = curInd[1]
        else:
            curVal = curVal[0]
            curInd = curInd[0]
        closestDict[i] = (curVal,curInd)

    with open('dict.pickle', 'wb') as handle:  # https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
        pickle.dump(closestDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return closestDict

def closest(clusters, using, closestDict, ind, removeInd):
    # closestDict tells us the previous closest cluster for each cluster -> (dist, ind)
    # removeInd is the index of the cluster we removed last time (-1 if first iteration)
    # ind is the index of the cluster we edited last time (contains also what used to be in clusters[removeInd]
    # using is the set of indices for clusters that contains a cluster that we are still using
    # clusters is a dictionary containing all the clusters, integer keys
    minVal = math.inf
    minInd = (0, 0)

    for i in using:
        if ind  != -1 and removeInd != -1:  # then there were changes made last time
            curVal, curInd = closestDict[i]  # curVal = dist to closest

            if curInd == removeInd  or curInd == ind:  # then we have to check everything to find the closest cluster
                minDist = math.inf
                newInd = 0
                for j in using:  # go through every cluster
                    if j != i:
                        newDist = 0
                        for d in range(len(cols) - 1):  # square distance
                            newDist += (clusters[j][1][d] - clusters[i][1][d]) ** 2
                        minDist = min(minDist, newDist)
                        if minDist == newDist:  # then this is the closest we have found so far
                            newInd = j
                closestDict[i] = (minDist, newInd)

            if i != ind:  #  then the closest cluster is either the same as last time, or clusters[ind] since every other center has not moved
                newDist = 0
                for d in range(len(cols)-1):
                    newDist += (clusters[i][1][d]-clusters[ind][1][d])**2
                if newDist < curVal:
                    closestDict[i] = (newDist, ind)
            else:  # then i==ind, this is the cluster whose center we have just moved, so check everything
                minDist = math.inf
                newInd = 0
                for j in using:  # go through every cluster
                    if j != ind:
                        newDist = 0
                        for d in range(len(cols) - 1):  # square distance
                            newDist += (clusters[j][1][d] - clusters[ind][1][d]) ** 2
                        minDist = min(minDist, newDist)
                        if minDist == newDist:  # then this is the closest we have found so far
                            newInd = j
                closestDict[ind] = (minDist, newInd)

        curVal, curInd = closestDict[i]
        minVal = min(curVal, minVal)
        if curVal == minVal:
            minInd = (i, curInd)
    return minInd[0], minInd[1], closestDict

def run_linkage_based_cluster(k, clusters, using, closestDict):
    lastChange = -1  # index of cluster that contains the combined cluster from the last iteration
    lastRemove = -1  # index of cluster that was remomved  in the last iteration
    for curK in range(len(clusters), k-1, -1):  # curK is the number of clusters we currently have
        print(curK)
        if curK == k:  # when we have k clusters (curK is the number of clusters we currently have)
            return clusters
        i, j, closestDict = closest(clusters, using, closestDict, lastChange, lastRemove)
        vals = []

        for d in range(len(cols)-1):
            vals.append((len(clusters[i][0])*clusters[i][1][d]+len(clusters[j][0])*clusters[j][1][d])/(len(clusters[i][0])+len(clusters[j][0])))  # weighted average of old coordinates for centers
        clusters[i][1] = tuple(vals)  # update center
        clusters[i][0]= clusters[i][0]+clusters[j][0]  # concatenate the two clusters
        clusters[j] = []
        using.remove(j)

        lastChange = i
        lastRemove = j
    return clusters

def plot(buckets, x, y):
    for k, v in buckets.items():
        if v != []:  # then we are using this cluster
            xCur = []
            yCur = []
            for i in v[0]:
                xCur.append(x[i])
                yCur.append(y[i])
            plt.scatter(xCur, yCur)

def write(clusters, points):
    f = open("clusters-linkage.txt", 'w')
    f.write("——————————————————————————————————————————————\n")
    for cluster in clusters.values():
        if cluster != []:  # if this is a cluster we are using
            for i in cluster[0]:
                f.write(points["movie_title"][i])
                f.write("\n")
            f.write("——————————————————————————————————————————————\n")
    f.close()

def linkage_based_cluster(fName, k):
    points = readFile(fName)
    clusters = {}
    for i in range(len(points[cols[0]])):
        cur = []
        for j in range(1, len(cols)):
            cur.append(points[cols[j]][i])
        clusters[i] = [[i], tuple(cur)]

    using = set(range(len(points[cols[0]])))  # using is the set of key's for clusters that actually map to a cluster we are using (as we merge clusters, we remove items from using)
    with open('dict.pickle', 'rb') as handle:
        closestDict = pickle.load(handle)  # calculated this before since it does not change
                                            # says which other cluster is closest to each cluster (used genClosestDict once and wrote result to dict.pickle)

    clusters = run_linkage_based_cluster(k, clusters, using, closestDict)
    write(clusters, points)

    return clusters, points

def makeGraph(clusters, C):
    lastMax = 0  # keep track of current y value
    minVal = math.inf  # minimum coefficient
    totalSum = 0  #  total sum of coefficient values
    totalNum = 0  # total number of points
    for i in range(len(C)):
        for j in range(len(C[i])):
            minVal = min(minVal, C[i][j])
            totalSum += C[i][j]
            totalNum += 1
    avg = totalSum/totalNum

    for i in range(len(clusters)):
        C[i].append(minVal)  # add this so that the fill is consistent between clusters
        C[i].sort()

        yV = [j+lastMax for j in range(len(C[i]))]  #  y values for this cluster
        lastMax += len(C[i])+1  # +1 so that there is a little space between each cluster
        plt.plot(C[i], yV)
        top = [lastMax-1 for i in  range(len(C[i]))]  # line at the top of the cluster
        plt.fill_between(C[i], yV, top)
    y = [i for i in range(lastMax)]  # y values for line showing mean
    x = [avg for i in range(lastMax)]  # x values are always just the mean since this is a vertical line
    plt.plot(x, y, color="maroon", linestyle="dotted")
    plt.yticks([]) # remove the values on the y axis since they mean nothing

def silhouette(fName, k):
    buckets, points = linkage_based_cluster(fName, k)
    clusters = []
    for i in range(len(buckets)):
        if buckets[i] != []:
            clusters.append(buckets[i][0])
    print(clusters)

    # plt.subplot(121)  # separate subplot for the graph of the points
    # plot(buckets, x, y)

    C = [[] for i in range(len(clusters))]
    for p in range(len(clusters)):
        for i in range(len(clusters[p])):
            A = 0
            B = math.inf
            for q in range(len(clusters)):
                if p != q:  # then we use this to calculate bi since the clusters are different
                    curB = 0
                    for j in range(len(clusters[q])):
                        for d in range(len(points)):
                            curB += (points[d][clusters[p][i]] - points[d][clusters[q][j]])**2
                    curB /= len(clusters[q])
                    B = min(curB, B)
                else:  # otherwise we are calculating ai
                    for j in range(len(clusters[p])):
                        for d in range(len(points)):
                            A += (points[d][clusters[p][i]] - points[d][clusters[q][j]])**2
                    A /= len(clusters[p])
            C[p].append((B-A)/max(A, B))
    # plt.subplot(122)
    makeGraph(clusters, C)
    plt.show()

linkage_based_cluster("newMovieData.csv", 100)
