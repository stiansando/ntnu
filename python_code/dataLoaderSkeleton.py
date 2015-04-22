__author__ = 'stiansando'
__author__ = 'cherdsakmangmee'
import Backprop_skeleton as Bp
from pylab import *

#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid]=[di]
        return dataset


def runRanker(trainingset, testset):
    #TODO: Insert the code for training and testing your ranker here.
    #Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = Bp.NN(46,10,0.001)

    #TODO: The lists below should hold training patterns in this format: [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    #TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    trainingPatterns = [] #For holding all the training patterns we will feed the network
    testPatterns = [] #For holding all the test patterns we will feed the network
    for qid in dhTraining.dataset.keys():
        #This iterates through every query ID in our training set
        dataInstance=dhTraining.dataset[qid] #All data instances (query, features, rating) for query qid
        #TODO: Store the training instances into the trainingPatterns array. Remember to store them as pairs, where the first item is rated higher than the second.
        #TODO: Hint: A good first step to get the pair ordering right, is to sort the instances based on their rating for this query. (sort by x.rating for each x in dataInstance)
        dataInstance.sort(key=lambda item: item.rating, reverse=True)
        for i in range(len(dataInstance)-1):
            for j in range(i+1, len(dataInstance)):
                if dataInstance[i].rating != dataInstance[j].rating:
                    trainingPatterns.append([dataInstance[i], dataInstance[j]])
                

    for qid in dhTesting.dataset.keys():
        #This iterates through every query ID in our test set
        dataInstance=dhTesting.dataset[qid]
        #TODO: Store the test instances into the testPatterns array, once again as pairs.
        #TODO: Hint: The testing will be easier for you if you also now order the pairs - it will make it easy to see if the ANN agrees with your ordering.
        dataInstance.sort(key=lambda item: item.rating, reverse=True)
        for i in range(len(dataInstance)-1):
            for j in range(i+1, len(dataInstance)):
                if dataInstance[i].rating != dataInstance[j].rating:
                    testPatterns.append([dataInstance[i], dataInstance[j]])

    test = []
    training = []
    #Check ANN performance before training
    print("Before training: ")
    print("     testset:")
    test.append(nn.countMisorderedPairs(testPatterns))
    print("     trainingset:")
    training.append(nn.countMisorderedPairs(trainingPatterns))
    for i in range(25):
        #Running 25 iterations, measuring testing performance after each round of training.
        #Training
        nn.train(trainingPatterns,iterations=1)
        print("Trainingiteration " + str(i+1))
        #Check ANN performance after training.
        print("     testset:")
        test.append(nn.countMisorderedPairs(testPatterns))
        print("     trainingset:")
        training.append(nn.countMisorderedPairs(trainingPatterns))

    return test, training


#----- M A I N -----
#TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.
overallTest = [0] * 26
overallTraining = [0] * 26

numOfRuns = 5
for i in range(numOfRuns):
    test, training = runRanker("../datasets/train.txt","../datasets/test.txt")
    for j in range(26):
        overallTest[j] += test[j]
        overallTraining[j] += training[j]

for i in range(26):
    overallTest[i] = overallTest[i]/5
    overallTraining[i] = overallTraining[i]/5    

plot(range(1,27), overallTest, label="test set")
plot(range(1,27), overallTraining, label="training set")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
ylim([0,1])
show()

