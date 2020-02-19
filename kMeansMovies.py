import pandas as pd
import numpy as np
from scipy import spatial
import math
import matplotlib.pyplot as plt

def findGenres(df):
    genresList = set() # all possible genres
    genresDict = {}  # genresDict[i] is a set of the genres of movie in i'th row of df

    for i in range(len(df["genres"])):  # go through genres column to make a list of genres (stored as a set), mark genres for each movie
        genres = df["genres"][i].split("|")
        for genre in genres:
            if genre not in genresList:  # used set so this is O(1)
                genresList.add(genre)
        genresDict[i] = set(genres)

    genresList = list(genresList)
    new_df = []  # list of lists to be turned into df, each column is one genre
                 # row i in new_df represents movie in row i of df, 1 means it is that genre, 0 means it is not
    for k, v in genresDict.items():
        cur = []
        for genre in genresList:
            if genre in v:  #  used set for v so this is O(1)
                cur.append(1)
            else:
                cur.append(0)
        new_df.append(cur)
    new_df = pd.DataFrame(new_df, index=[i for i in range(len(df))],columns=genresList)
    return new_df

def readFile(fname):
    global cols

    df = pd.read_csv(fname)
    cols = list(df.columns.values)
    del df[cols[0]]
    cols = cols[1:]
    return df

def modifyData():
    df = pd.read_csv("movieData.csv")

    df = df.dropna()  # remove rows with missing data

    df = df[df["country"].isin(["USA","UK"])] # only want US/UK movies

    df["country"] = df["country"].replace("USA", 1)
    df["country"] = df["country"].replace("UK", 0)
    df = df.reset_index()
    del df["index"]
    del df["aspect_ratio"]

    df = df[df.content_rating != "Unrated"]  # remove movies with weird ratings
    df = df[df.content_rating != "Not Rated"]
    df = df[df.content_rating != "GP"]
    df = df[df.content_rating != "Passed"]
    df = df[df.content_rating != "Approved"]
    df = df[df.content_rating != "M"]

    df["content_rating"] = df["content_rating"].replace("G", 0)
    df["content_rating"] = df["content_rating"].replace("PG", 1)
    df["content_rating"] = df["content_rating"].replace("PG-13", 2)
    df["content_rating"] = df["content_rating"].replace("R", 3)
    df["content_rating"] = df["content_rating"].replace("NC-17", 4)
    df["content_rating"] = df["content_rating"].replace("X", 5)

    df = df.reset_index()
    del df["index"]

    df2 = findGenres(df)  # df2 is dataframe of genres, 1 if movie is that genre
    del df["genres"]
    df = df.reset_index()

    df = pd.concat([df, df2],axis=1)
    del df["index"]

    j=0
    for (n, d) in df.iteritems():  # STANDARDIZE
        print(j)
        j=j+1
        if n != "movie_title":  # then this column has numbers
            minV = min(d)
            maxV = max(d)
            for i in range(len(d)):  # make all values between 0 and 1
                df.ix[i, n] = (d[i] - minV)/(maxV-minV)

    df.to_csv("newMovieData.csv")

def findClosest(curLoc, points):
    indices = []
    tree = spatial.KDTree(curLoc)

    for i in range(len(points[cols[0]])):  # go through every point and find the closest cluster
        row = points.ix[[i]]
        new_row = [row[cols[j]][i] for j in range(1, len(cols))]
        indices.append(tree.query(new_row, 1)[1])
    return indices

def totalDist(curLoc, points):
    total = 0
    tree = spatial.KDTree(curLoc)

    for i in range(len(points[cols[0]])):  # go through every point and find the closest cluster
        row = points.ix[[i]]
        new_row = [row[cols[j]][i] for j in range(1, len(cols))]
        total += tree.query(new_row, 1)[0]
    return total

def find_center(bucket, points):
    tot = [0  for i in range(len(cols)-1)]
    for i in bucket:
        for a in range(1, len(cols)):
            d = cols[a]
            tot[a-1] += points[d][i]

    return [tot[i]/len(bucket) for i in range(len(tot))]  # average of x and y coordinates for every point in the cluster

def run_kMeans(curLoc, points, old_buckets, k, count=0):
    print(count)
    indices = findClosest(curLoc, points)  # indices[i] gives the closest center to (x[i], y[i]) (indices[i] = j means curLoc[j] is closest)
    buckets = {}  # maps j to a list where j represents curLoc[j] and the list is its cluster
    for i in range(len(indices)):
        if indices[i] in buckets:
            buckets[indices[i]].append(i)
        else:
            buckets[indices[i]] = [i]

    if old_buckets == buckets:
        return curLoc, buckets

    centers = []  # to store new locations of centers

    for val in buckets.values():
        centers.append(find_center(val, points))

    return run_kMeans(centers, points, buckets, k, count+1)

def randomSol(k, points):
    randoms = np.random.randint(0, len(points[cols[0]]), k)  # choose random centers as initial solution
    curLoc = []
    for random in randoms:
        cur = []
        for (key, item) in points.iteritems():
            if  key != "movie_title":   # then this is a number
                cur.append(item[random])
        curLoc.append(cur)

    return curLoc

def plot(buckets, centers, x, y):  # only for 2D
    for k, v in buckets.items():  # graph the clusters
        xCur = []
        yCur = []
        for i in v:
            xCur.append(x[i])
            yCur.append(y[i])
        plt.scatter(xCur, yCur)
    for item in centers:
        plt.plot(item[0], item[1], color='r', marker='x')
    plt.show()

def kMeans(fName, k):
    points = readFile(fName)
    curLoc = randomSol(k, points)
    centers, buckets = run_kMeans(curLoc, points, {}, k)
    write(buckets, points)
    return buckets, centers, points

def elbow(fName):
    ind = [i for i in range(10, 101, 10)]   # we just look at values between 1 and 10
    total = []
    for k in range(10, 101, 10):
        clusters, centers, points = kMeans(fName, k)
        total.append(totalDist(centers, points))
        print(k, total[-1])
    plt.plot(ind, total)
    plt.savefig("plot.png")
    plt.show()
    plt.close()

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

def write(clusters, points):
    f = open("clusters-k-means.txt", 'w')
    f.write("——————————————————————————————————————————————\n")
    for cluster in clusters.values():
        for i in cluster:
            f.write(points["movie_title"][i])
            f.write("\n")
        f.write("——————————————————————————————————————————————\n")
    f.close()

def silhouette(fName, k):
    clusters, centers, points = kMeans(fName,k)
    # plt.subplot(121)  # separate subplot for the graph of the points
    # plot(clusters, centers, points[0], points[1])

    for cluster in clusters.values():
        print("——————————————————————————————————————————————")
        for i in cluster:
            print(i)
            for j in range(1, len(cols)):
                pass
    quit()

    C = [[] for i in range(len(clusters))]
    for p in range(len(clusters)):
        print(p)
        for i in range(len(clusters[p])):
            A = 0
            B = math.inf
            for q in range(len(clusters)):
                if p != q:  # then we use this to calculate bi since the clusters are different
                    curB = 0
                    for j in range(len(clusters[q])):
                        for a in range(1, len(cols)):
                            d = cols[a]
                            curB += (points[d][clusters[p][i]] - points[d][clusters[q][j]])**2
                    curB /= len(clusters[q])
                    B = min(curB, B)
                else:  # otherwise we are calculating ai
                    for j in range(len(clusters[p])):
                        for a in range(1, len(cols)):
                            d = cols[a]
                            A += (points[d][clusters[p][i]] - points[d][clusters[q][j]])**2
                    A /= len(clusters[p])
            C[p].append((B-A)/max(A, B))
    # plt.subplot(122)
    makeGraph(clusters, C)
    plt.savefig("plot-silhouette.png")

kMeans("newMovieData.csv", 40)
# modifyData()
