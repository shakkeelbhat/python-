{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv\n",
    "import math\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "def loadCsv(filename):\n",
    "    lines = csv.reader(open(r'C:\\Users\\onlyp\\Documents\\SUBLIME_TEXT_SAVES\\diabetes.csv'))\n",
    "    dataset = list(lines)\n",
    "    for i in range(len(dataset)):\n",
    "        dataset[i] = [float(x) for x in dataset[i]]\n",
    "    return dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "def splitDataset(dataset, splitRatio):\n",
    "    trainSize = int(len(dataset) * splitRatio)\n",
    "    trainSet = []\n",
    "    copy = list(dataset)\n",
    "    while len(trainSet) & lt in trainSize:\n",
    "        index = random.randrange(len(copy))\n",
    "        trainSet.append(copy.pop(index))\n",
    "    return [trainSet, copy]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "def separateByClass(dataset):\n",
    "    separated = {}\n",
    "    for i in range(len(dataset)):\n",
    "        vector = dataset[i]\n",
    "    if (vector[-1] not in separated):\n",
    "        separated[vector[-1]] = []\n",
    "        separated[vector[-1]].append(vector)\n",
    "    return separated"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mean(numbers):\n",
    "    return sum(numbers)/float(len(numbers))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "def stdev(numbers):\n",
    "    avg = mean(numbers)\n",
    "    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)\n",
    "    return math.sqrt(variance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "def summarize(dataset):\n",
    "    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]\n",
    "    del summaries[-1]\n",
    "    return summaries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "def summarizeByClass(dataset):\n",
    "    separated = separateByClass(dataset)\n",
    "    summaries = {}\n",
    "    for classValue, instances in separated.items():\n",
    "        summaries[classValue] = summarize(instances)\n",
    "    return summaries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculateProbability(x, mean, stdev):\n",
    "    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))\n",
    "    return (1/(math.sqrt(2*math.pi)*stdev))*exponent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculateClassProbabilities(summaries, inputVector):\n",
    "    probabilities = {}\n",
    "    for classValue, classSummaries in summaries.items():\n",
    "        probabilities[classValue] = 1\n",
    "    for i in range(len(classSummaries)):\n",
    "        mean, stdev = classSummaries[i]\n",
    "        x = inputVector[i]\n",
    "        probabilities[classValue] *= calculateProbability(x, mean, stdev)\n",
    "    return probabilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(summaries, inputVector):\n",
    "    probabilities = calculateClassProbabilities(summaries, inputVector)\n",
    "    bestLabel, bestProb = None, -1\n",
    "    for classValue, probability in probabilities.items():\n",
    "        if bestLabel is None or probability & bestProb:\n",
    "            bestProb = probability\n",
    "            bestLabel = classValue\n",
    "    return bestLabel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "def getPredictions(summaries, testSet):\n",
    "    predictions = []\n",
    "    for i in range(len(testSet)):\n",
    "        result = predict(summaries, testSet[i])\n",
    "        predictions.append(result)\n",
    "    return predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "def getAccuracy(testSet, predictions):\n",
    "    correct = 0\n",
    "    for x in range(len(testSet)):\n",
    "        if testSet[x][-1] == predictions[x]:\n",
    "            correct += 1\n",
    "    return (correct/float(len(testSet)))*100.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'Pregnancies'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-67-13d9c7a9a8d0>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m     11\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Accuracy: {0}%'\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maccuracy\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 13\u001b[1;33m \u001b[0mmain\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-67-13d9c7a9a8d0>\u001b[0m in \u001b[0;36mmain\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mmain\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m     \u001b[0msplitRatio\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m0.67\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m     \u001b[0mdataset\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mloadCsv\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m     \u001b[0mtrainingSet\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtestSet\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msplitDataset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msplitRatio\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Split {0} rows into train = {1} and test = {2} rows'\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrainingSet\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtestSet\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-37-57f47b0fa122>\u001b[0m in \u001b[0;36mloadCsv\u001b[1;34m(filename)\u001b[0m\n\u001b[0;32m      3\u001b[0m     \u001b[0mdataset\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlines\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m         \u001b[0mdataset\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mfloat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mdataset\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mdataset\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-37-57f47b0fa122>\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m      3\u001b[0m     \u001b[0mdataset\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlines\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m         \u001b[0mdataset\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mfloat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mdataset\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mdataset\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: could not convert string to float: 'Pregnancies'"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    splitRatio = 0.67\n",
    "    dataset = loadCsv(filename)\n",
    "    trainingSet, testSet = splitDataset(dataset, splitRatio)\n",
    "    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))\n",
    "#prepare model\n",
    "    summaries = summarizeByClass(trainingSet)\n",
    "#test model\n",
    "    predictions = getPredictions(summaries, testSet)\n",
    "    accuracy = getAccuracy(testSet, predictions)\n",
    "    print('Accuracy: {0}%'.format(accuracy))\n",
    " \n",
    "main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
