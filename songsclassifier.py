import pandas as pd
import numpy as np
import math


def manhattan1(rating1,rating2):
    return sum(abs((rating1-rating2).dropna()))


def computeNearestNeighborSongs(itemName,itemAttr,music):
   # music1 = music.drop([itemName],axis=1)
    return sorted(music.columns.map(lambda user: (manhattan1(music[user],itemAttr),user)).tolist(),key=lambda at:at[0])

def classify(user,itemName,itemSeries,users,music):
    nearest = computeNearestNeighborSongs(itemName,itemSeries,music)[0][1]
    return users[user][nearest]

def getMedian(dataSeries):
    return dataSeries.median()

def getAbsoluteStandardDeviation(dataSeries):
    median = getMedian(dataSeries)
    return (dataSeries.map(lambda x: abs(x-median)).sum())/(len(dataSeries))

def normalizeColumn(column,data):
    median = getMedian(data[column])
    asd = getAbsoluteStandardDeviation(data[column])
    #asd = data[column].std()
    return [ (x-median)/asd for x in data[column]]

def normalizeInput(list,data):
    hmedian = getMedian(data["Height"])
    hasd = getAbsoluteStandardDeviation(data["Height"])
    wmedian = getMedian(data["Weight"])
    wasd = getAbsoluteStandardDeviation(data["Weight"])
    return [(list[0]-hmedian)/hasd,(list[1]-wmedian)/wasd]


def computeNearestNeighborW(list,data):
   return  data.index.map(lambda x: [sum(map(lambda u,v: abs(u-v),list,[data.ix[x]["normalizedHeight"],data.ix[x]["normalizedWeight"]])),data.ix[x]["comment"],data.ix[x]["class"]])

def computeNearestNeighborW1(list,data):
   return  data.index.map(lambda x: [math.sqrt(sum(map(lambda u,v: pow(u-v,2),list,[data.ix[x]["normalizedHeight"],data.ix[x]["normalizedWeight"]]))),data.ix[x]["comment"],data.ix[x]["class"]])
