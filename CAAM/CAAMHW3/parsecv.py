import pandas as pd
import math
import matplotlib

def distance(tuple1, tuple2):
    deltax = abs(tuple1[0] - tuple2[0])
    deltay = abs(tuple1[1] - tuple2[1])
    return math.sqrt(pow(deltax, 2) + pow(deltay, 2))

def k_means(df, guesses, threshold): 
    change = 1
    temp = 0
    
    while change > threshold:
        #print("CHANGE: ", change)
        change = 0
        clusters = []
        for i in range(len(guesses)):
            clusters.append([])
        #Set Clusters:
        for i, row in df.iterrows(): #For each entry, we need to calculate distance, and check to the different clusters
            index = -1
            minInd = -1
            #print(row)
            min = 1000000
            for center in guesses:
                index += 1
                dist = distance(center, row) # This should calculate distance
                if min > dist:
                    min = dist
                    minInd = index

            clusters[minInd].append(row) 
        #Update Guesses
        #print(guesses)
        #for cluster in clusters:
            #print(len(cluster))
        #Find Average Coordinates for cluster
        for inds in range(len(clusters)):
            sumX = 0
            sumY = 0
            for entry in clusters[inds]:
                sumX += entry[0]
                sumY += entry[1]
            newY = sumY/len(clusters[inds])
            newX = sumX/len(clusters[inds])
            change += distance(center, (newX, newY))
            guesses[inds][0] = newX
            guesses[inds][1] = newY
            temp +=1
        if temp > 20:
            #print(temp)
            break
    return guesses

df = pd.read_csv('CAAM/CAAMHW3/raw_uber_data.csv', usecols=['Lat','Lon'])
df1 = pd.read_csv('CAAM/CAAMHW3/raw_uber_data_weekday_afternoons.csv', usecols=['Lat','Lon'])
df2 = pd.read_csv('CAAM/CAAMHW3/raw_uber_data_weekday_evenings.csv', usecols=['Lat','Lon'])
df3 = pd.read_csv('CAAM/CAAMHW3/raw_uber_data_weekday_mornings.csv', usecols=['Lat','Lon'])
guesses = [[40.77, -73.9], [40.78, -73.88], [40.81, -73.94], [40.8, -74], [40.9, -74], [40.9, -73.87]]
guesses1 = [[40.77, -73.9], [40.78, -73.88], [40.81, -73.94], [40.8, -74], [40.9, -74], [40.9, -73.87]]
guesses2 = [[40.77, -73.9], [40.78, -73.88], [40.81, -73.94], [40.8, -74], [40.9, -74], [40.9, -73.87]]
guesses3 = [[40.77, -73.9], [40.78, -73.88], [40.81, -73.94], [40.8, -74], [40.9, -74], [40.9, -73.87]]
ansk = k_means(df, guesses, 0.1)
ans1 = k_means(df1, guesses1, 0.1)
ans2 = k_means(df2, guesses2, 0.1)
ans3 = k_means(df3, guesses3, 0.1)

temp = [ansk, ans1, ans2, ans3]
descriptor = 0
for ans in temp:
    if descriptor == 0:
        print("The initial dataset:")
    elif descriptor == 1:
        print("Weekday Mornings:")
    elif descriptor == 2:
        print("Weekday Afternoons:")
    else:
        print("Weekday Evenings: ")
    ind = 0
    descriptor += 1
    for res in ans:
        ind += 1
        print("The ", ind , "th optimal location is: ", res)
    print("")
#print(df)