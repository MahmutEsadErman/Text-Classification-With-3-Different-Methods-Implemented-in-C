#include <stdio.h>
#include <math.h>
#include <time.h>
#include "functionsForDictionaryAndHotVectors.h"

#define OPFUNC 2
#define STARTW 0

void setValuesToArray(double* arr, double value, int length);
double gradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors);
double stochasticGradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors,int stoch_size);
double ADAM(int maxIter, int exampleNumber, double learning_rate, int dictlength,double* weights,bool ** hotvectors,double b1 ,double b2 ,int stoch_size);

double test(bool** hotVectors, double* weights, int dictLength, int exampleNumber);
void saveStatistics(char* path, double loss, double succesRate, double duration);

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
    char* outputFilePath = "dataset/statistics.txt";
    bool** trainingHotVectors;
    bool** testHotVectors;
    double* weights;
    clock_t start_time, end_time;
    FILE* group1TrainingFile;
    FILE* group2TrainingFile;
    FILE* group1TestFile;
    FILE* group2TestFile;

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
    setValuesToArray(weights, STARTW, dictionary.length);
    double step_size = 0.01;
    int epochs = 100;
    double B1= 0.9;
    double B2= 0.99;
    int stoch_size = 10;
    start_time = clock();

    switch (OPFUNC)
    {
    case 0:
        gradientDescent(epochs, numberOfTrainingExamples, step_size, dictionary.length, weights, trainingHotVectors);
        break;
    case 1:
        stochasticGradientDescent(epochs, numberOfTrainingExamples, step_size, dictionary.length, weights, trainingHotVectors,stoch_size);
        break;
    case 2:
        ADAM(epochs, numberOfTrainingExamples, step_size, dictionary.length, weights, trainingHotVectors,B1 ,B2,stoch_size);
    default:
        break;
    }

    printf("duration: %.2f\n", (double)(clock()-start_time) / CLOCKS_PER_SEC * 1000);

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
    double loss,gt;
    double *dp;
    dp = malloc(exampleNumber*sizeof(double));
    for (epoch = 0; epoch < maxIter; epoch++) {
        loss = 0;
        
        
        for (i = 0; i < exampleNumber; i++) {
            dp[i] = dotProduct(weights, hotVectors[i], dictLength);
        }
        for (j = 0; j < dictLength; j++) {
            gt=0;
            y = -1;
            for (i = 0; i < exampleNumber; i++) {
                if(y == 1) y = -1;
                else y = 1;
                gt+=lossFunction_der(dp[i], y, hotVectors[i][j]);
            }
            gt/=exampleNumber;
            // Gradient Descent
            weights[j] -= learning_rate * gt;
        }
        y=-1;
        for(i=0;i<exampleNumber; i++){
            if(y == 1) y = -1;
            else y = 1;
            dp[i] = dotProduct(weights, hotVectors[i], dictLength);
            loss += lossFunction(y, function(dp[i]));
        }
        loss /= exampleNumber;
        printf("Epoch: %d, Loss: %f\n", epoch+1, loss);
    }
    return loss;
}

double stochasticGradientDescent(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors,int stoch_size){
    int i,j,epoch,y;
    double loss;
    int start, end;
    double *dp;
    int * random_num;
    random_num = malloc(stoch_size*sizeof(int));
    dp = malloc(exampleNumber*sizeof(double));
    for (epoch = 0; epoch < maxIter; epoch++) {
        loss = 0;
        
        for (i=0; i<stoch_size; i++){
            random_num[i] = rand() % exampleNumber;
            dp[i] = dotProduct(weights, hotVectors[random_num[i]], dictLength);
        }
        for (j = 0; j < dictLength; j++) {
            for (i = 0; i < stoch_size; i++) {
                y= (random_num[i]%2) *(-2) +1 ;
                // SGradient Descent
                weights[j] -= learning_rate * lossFunction_der(dp[i], y, hotVectors[random_num[i]][j]);
            }
        }
        for(i=0;i<stoch_size; i++){
            y= (random_num[i]%2) *(-2) +1 ;
            dp[i] = dotProduct(weights, hotVectors[random_num[i]], dictLength);
            loss += lossFunction(y, function(dp[i]));
        }
        loss /= exampleNumber;
        printf("Epoch: %d, Loss: %f\n", epoch+1, loss);
    }
    return loss;
}

double ADAM(int maxIter, int exampleNumber, double learning_rate, int dictLength, double* weights, bool** hotVectors, double b1,double b2,int stoch_size){
    int i,j,epoch,y;
    double loss;
    double * dp;
    double *mt,*vt,gt,eps=0.0000001;
    int  * random_num;
    random_num = malloc(stoch_size*sizeof(int));
    dp = malloc(stoch_size*sizeof(double));
    mt = calloc(dictLength,sizeof(double));
    vt= calloc(dictLength,sizeof(double));
    for (epoch = 0; epoch < maxIter; epoch++) {
        
        loss = 0;
        for (i=0; i<stoch_size; i++){
            random_num[i] = rand() % exampleNumber;
            dp[i] = dotProduct(weights, hotVectors[random_num[i]], dictLength);
        }
        for (j = 0; j < dictLength; j++) {
            for (i = 0; i < stoch_size; i++) {
                y= (random_num[i]%2) *(-2) +1 ;
                
                gt+=lossFunction_der(dp[i], y, hotVectors[random_num[i]][j]);
                
            }
            gt /= stoch_size;
            // ADAM
            mt[j]=mt[j]*b1 + (1-b1) * gt;
            vt[j]= vt[j]*b2 + (1-b2) * gt * gt;
            weights[j] -= learning_rate * (mt[j]/(1-pow(b1,epoch+1))) / pow((vt[j]/(1-pow(b2,epoch+1)) )+ eps,0.5);
            gt=0; 
        }
       
        for (i=0; i<stoch_size; i++){
            y= (random_num[i]%2) *(-2) +1 ;
            dp[i] = dotProduct(weights, hotVectors[random_num[i]], dictLength);
            loss += lossFunction(y, function(dp[i]));
        }
        loss /= stoch_size;
        printf("Epoch: %d, Loss: %f\n", epoch+1, loss);
    }
    free(dp);
    free(random_num);
    free(mt);
    free(vt);
    return loss;
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

void saveStatistics(char* path, double loss, double succesRate, double duration){

}

