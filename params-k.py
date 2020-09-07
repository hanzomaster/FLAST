from sklearn.model_selection import ShuffleSplit
import numpy as np
import os
import time

import flast


if __name__ == "__main__":
    print("MSR COMPARISON - SAMPLED DATASET")

    outDir = "results/params-k"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    projectBasePath = "dataset"
    projectName = "pinto-ds"

    numKFold = 30
    testSize = 0.1

    ###########################################################################
    # FLAST
    dim = -1  # number of dimensions (-1: no reduction; 0: JL with error eps)
    eps = 0
    reduceDim = False if dim < 0 else True  # reduce dimensionality if True
    threshold = 0.5

    # data points vectorization
    v0 = time.perf_counter()
    dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    Z = flast.flastVectorization(dataPoints, reduceDim=reduceDim, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
    v1 = time.perf_counter()
    vecTime = v1 - v0

    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        params = {
            "algorithm": "brute",
            "metric": "cosine",
            "k": k,
            "weights": "distance",
        }

        # prepare output folder
        print()
        print(f"K: {k}; Threshold: {threshold}; Eps: {eps}")
        outputFile = f"{outDir}/FLAST___k{k}.csv"
        with open(outputFile, "w") as fileOut:
            fileOut.write("fold,numFlakyTrainSet,numNonFlakyTrainSet,numFlakyTestSet,numNonFlakyTestSet,vecTime,trainTime,testTime,avgPredTime,f-measure,precision,recall,accuracy,tp,fp,fn,tn\n")

        kf = ShuffleSplit(n_splits=numKFold, test_size=testSize)
        successFold = 0
        for kFold, (trnIdx, tstIdx) in enumerate(kf.split(dataPointsList, dataLabelsList)):
            trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
            trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
            if sum(trainLabels) == 0 or sum(testLabels) == 0:
                print("Skipping fold...")
                print(" Flaky Train Tests", sum(trainLabels))
                print(" Flaky Test Tests", sum(testLabels))
                continue
            else:
                successFold += 1

            # prepare the data in the right format for kNN
            nSamplesTrainData, nxTrain, nyTrain = trainData.shape
            trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
            nSamplesTestData, nxTest, nyTest = testData.shape
            testData = testData.reshape((nSamplesTestData, nxTest * nyTest))

            trainTime, testTime, predictLabels = flast.flastClassification(trainData, trainLabels, testData, threshold, params)
            res = flast.computeResults(trainData, testData, trainLabels, testLabels, predictLabels)
            print(f"Precision: {res['precision']}")
            print(f"Recall: {res['recall']}")

            with open(outputFile, "a") as fileOut:
                fileOut.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        successFold,
                        sum(trainLabels),
                        len(trainLabels) - sum(trainLabels),
                        sum(testLabels),
                        len(testLabels) - sum(testLabels),
                        vecTime,
                        trainTime,
                        testTime,
                        testTime / len(testData),
                        res["f-measure"],
                        res["precision"],
                        res["recall"],
                        res["accuracy"],
                        res["tp"],
                        res["fp"],
                        res["fn"],
                        res["tn"]
                    )
                )
