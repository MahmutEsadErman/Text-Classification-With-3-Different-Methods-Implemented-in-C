cmake_minimum_required(VERSION 3.26)
project(textClassification C)

set(CMAKE_C_STANDARD 17)

add_executable(homework1 main.c
        sortingArticlesbyTopics.c
        makeDictionaryAndHotVectors.c)
