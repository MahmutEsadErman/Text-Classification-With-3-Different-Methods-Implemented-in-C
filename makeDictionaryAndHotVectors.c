#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>

#define NUMBEROFWORDS 7000

// Struct to hold each key-value pair
struct KeyValuePair {
    char key[50];
    int value;
};

// Dictionary structure
struct Dictionary {
    struct KeyValuePair items[NUMBEROFWORDS]; // Defined as an array with a capacity of 100, can be adjusted as needed
    int length; // Keeps track of the number of items in the dictionary
}typedef dict;

// makes letters short
void toLowerStr(char *p){
    for ( ; *p; ++p) *p = tolower(*p);
}

void addToDictionary(dict* dictionary, const char* key, int value);
int getValue(dict* dictionary, const char* key);
void saveDictionary(FILE* file, dict* dictionary);
void saveVector(FILE* file, bool* vector, int n);
void makeDictionary(FILE* file, dict* dictionary);
void makeHotVector(dict generalDict, dict topicDict, bool* hotVector);

int main() {
    dict generalDictionary;
    dict businessDictionary;
    dict sportDictionary;
    generalDictionary.length = 0;
    businessDictionary.length = 0;
    sportDictionary.length = 0;

    bool** businessVectors;
    bool** sportVectors;

    FILE* businessTraining;
    FILE* sportTraining;
    FILE* dictionaryOutput;
    FILE* businessOutput;
    FILE* sportOutput;

    businessTraining = fopen("dataset/businessTraining.txt", "r");
    sportTraining = fopen("dataset/sportTraining.txt", "r");
    dictionaryOutput = fopen("dataset/dictionary.txt", "w");
    businessOutput = fopen("dataset/businessVector.txt", "w");
    sportOutput = fopen("dataset/sportVector.txt", "w");

    // make general dictionary
    makeDictionary(businessTraining, &generalDictionary);
    makeDictionary(sportTraining, &generalDictionary);

    // make business dataset

    // make sport dataset


    // saveFiles
    saveDictionary(dictionaryOutput, &generalDictionary);
    //saveVector(businessOutput, businessVectors, generalDictionary.length);
    //saveVector(sportOutput, sportVectors, generalDictionary.length);

    // Freeing memory
    free(businessVectors);
    free(sportVectors);
    fclose(businessTraining);
    fclose(sportTraining);
    fclose(dictionaryOutput);
    fclose(businessOutput);
    fclose(sportOutput);

    printf("Dictionary and Vectors are saved ");

    return 0;
}


void addToDictionary(dict* dictionary, const char* key, int value) {
    if (dictionary->length < NUMBEROFWORDS) { // If there is space in the dictionary
        strcpy(dictionary->items[dictionary->length].key, key);
        dictionary->items[dictionary->length].value = value;
        dictionary->length++;
    } else {
        printf("Dictionary is full, cannot add new item.\n");
    }
}

// Function to get the value for a given key in the dictionary
int getValue(dict* dictionary, const char* key) {
    for (int i = 0; i < dictionary->length; i++) {
        if (strcmp(dictionary->items[i].key, key) == 0) {
            return dictionary->items[i].value;
        }
    }
    return -1; // Return -1 if the key is not found (default value)
}

// Saves dictionary to a txt file
void saveDictionary(FILE* file, dict* dictionary){
    int i;
    for (i = 0; i < dictionary->length; i++){
        if(strcmp(dictionary->items[i].key, "\0")  != 0){
            fprintf(file, "%s\n", dictionary->items[i].key);
        } else{
            printf("%d)\n", i);
        }
    }
}

// Saves vector into files
void saveVector(FILE* file, bool* vector, int n){
    int i;
    for (i = 0; i < n; i++){
        fprintf(file, "%d\n", vector[i]);
    }
}

void makeDictionary(FILE* file, dict* dictionary){
    char word[50];
    rewind(file);
    while (fscanf(file, "%s", word) == 1){
        toLowerStr(word);
        if(getValue(dictionary, word) == -1){
            addToDictionary(dictionary, word, 1);
            if(dictionary->length == NUMBEROFWORDS){
                printf("you have reached word limit!!");
                return;
            }
        }
    }
}

void makeHotVector(dict generalDict, dict topicDict, bool* hotVector){
    int i;
    for (i = 0; i < generalDict.length; i++){
        if(getValue(&topicDict,generalDict.items[i].key) != -1){
            hotVector[i] = 1;
        }
    }
}