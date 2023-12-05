#include <stdio.h>
#include <math.h>
#include <time.h>
#include "loadDictionaryAndMakeHotVector.h"

void setValuesToArray(double* arr, double value, int length);
double gradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors);
double stochasticGradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors);

double test(bool** hotVectors, double* weights, int dictLength, int exampleNumber);

double function(double wx){
    return tanh(wx);
}

double function_der(double wx, bool x){
    return x*(1 - pow((tan(wx)), 2));
}

// Minimum Squared Error
double lossFunction(double y, double yPredicted){
    return pow((y-yPredicted), 2);
}
double lossFunction_der(double dotproduct, double y, bool x){
    return 2*(y - function(dotproduct)) * function_der(dotproduct, x);
}

double dotProduct(const double* w, const bool* x, int n){
    int i; double sum=0;
    for(i=0; i<n; i++) sum += w[i]*x[i];
    return sum;
}

int main() {
    int i, numberOfTrainingExamples, numberOfTestExamples;
    dict dictionary;
    char* group1_training_filePath = "dataset/businessTraining.txt";
    char* group2_training_filePath = "dataset/sportTraining.txt";
    char* group1_test_filePath = "dataset/businessTest.txt";
    char* group2_test_filePath = "dataset/sportTest.txt";
    bool** trainingHotVectors;
    bool** testHotVectors;
    double* weights;
    FILE* group1TrainingFile;
    FILE* group2TrainingFile;
    FILE* group1TestFile;
    FILE* group2TestFile;

    srand(time(NULL));

    // Loading Dictionary
    loadDictionary("dataset/dictionary.txt", &dictionary);
    printf("%s\n", "Dictionary is loaded! ");

    // Initializing files
    group1TrainingFile = fopen(group1_training_filePath, "r");
    group2TrainingFile = fopen(group2_training_filePath, "r");
    group1TestFile = fopen(group1_test_filePath, "r");
    group2TestFile = fopen(group2_test_filePath, "r");

    // Initializing weights
    weights = (double *) calloc(dictionary.length, sizeof(double));

    // Initializing training datasets, weights and files
    numberOfTrainingExamples = calculateLineNumber(group1_training_filePath)+calculateLineNumber(group2_training_filePath);
    trainingHotVectors = (bool **) calloc(numberOfTrainingExamples, sizeof(bool*));

    // Çift sayılara grup 1, Tek sayılara grup 2
    for (i = 0; i < numberOfTrainingExamples; i++) {
        trainingHotVectors[i] = (bool *) calloc(dictionary.length, sizeof(bool));
        if(i%2 == 0){
            makeHotVector(group1TrainingFile, dictionary, trainingHotVectors[i]);
        }else{
            makeHotVector(group2TrainingFile, dictionary, trainingHotVectors[i]);
        }
    }

    // Initializing test datasets
    numberOfTestExamples = calculateLineNumber(group1_test_filePath)+calculateLineNumber(group2_test_filePath);
    testHotVectors = (bool **) calloc(numberOfTrainingExamples, sizeof(bool*));

    // Çift sayılara grup 1, Tek sayılara grup 2
    for (i = 0; i < numberOfTestExamples; i++) {
        testHotVectors[i] = (bool *) calloc(dictionary.length, sizeof(bool));
        if(i%2 == 0){
            makeHotVector(group1TestFile, dictionary, testHotVectors[i]);
        }else{
            makeHotVector(group2TestFile, dictionary, testHotVectors[i]);
        }

    }

    //Training
    setValuesToArray(weights, 0, dictionary.length);
    double step_size = 0.01;
    int epochs = 100;
    int startTime = (int)time(NULL);

    //gradientDescent(epochs, numberOfTrainingExamples, step_size, dictionary.length, weights, trainingHotVectors);

    stochasticGradientDescent(epochs, numberOfTrainingExamples, step_size, dictionary.length, weights, trainingHotVectors);

    printf("duration: %d\n", (int)time(NULL)-startTime);

    // Testing
    test(testHotVectors, weights, dictionary.length, numberOfTestExamples);

    // Closing and Freeing
    fclose(group1TrainingFile);
    fclose(group2TrainingFile);
    free(trainingHotVectors);
    free(testHotVectors);
    free(weights);
}

void setValuesToArray(double* arr, double value, int length){
    int i;
    for(i = 0; i<length; i++){
        arr[i] = value;
    }
}

double gradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors){
    int i,j,epoch, y;
    double loss;
    double dp;
    for (epoch = 0; epoch < maxIter; epoch++) {
        loss = 0;
        y = -1;
        for (i = 0; i < exampleNumber; i++) {
            dp = dotProduct(weights, hotVectors[i], dictLength);
            if(y == 1) y = -1;
            else y = 1;
            // Gradient Descent
            for (j = 0; j < dictLength; j++) {
                weights[j] -= learning_rate * lossFunction_der(dp, y, hotVectors[i][j]);
            }
            loss += lossFunction(y, function(dp));
        }
        loss /= exampleNumber;
        printf("Epoch: %d, Loss: %f\n", epoch+1, loss);
    }
    return loss;
}

double stochasticGradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors){
    int i,j,epoch,y;
    double loss;
    double dp;
    int start, end;
    for (epoch = 0; epoch < maxIter; epoch++) {
        end = rand() % dictLength+1;
        start = rand() % end+1;
        loss = 0;
        y = -1;
        for (i = 0; i < exampleNumber; i++) {
            dp = dotProduct(weights, hotVectors[i], dictLength);
            if(y == 1) y = -1;
            else y = 1;
            // Gradient Descent
            for (j = start; j < end; j++) {
                weights[j] -= learning_rate * lossFunction_der(dp, y, hotVectors[i][j]);
            }
            loss += lossFunction(y, function(dp));
        }
        loss /= exampleNumber;
        printf("Epoch: %d, Loss: %f\n", epoch+1, loss);
    }
    return loss;
}

double ADAM(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors, int y){
    return 0;
}

double test(bool** hotVectors, double* weights, int dictLength, int exampleNumber){
    double successRate, y;
    int success = 0;
    int i;
    for (i = 0; i < exampleNumber; i++) {
        y = function(dotProduct(weights, hotVectors[i], dictLength));
        if(i % 2 == 0){
            if(y > 0) success++;
        } else{
            if(y < 0) success++;
        }
    }
    successRate = (double)success / exampleNumber;
    printf("Your model has correctly predicted %d words out of %d\n", success, exampleNumber);
    printf("Your model is %.2f percent successful!!\n", successRate*100);
    return successRate;
}