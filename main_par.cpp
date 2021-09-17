#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include <semaphore.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include "bits_file/stdc++.h"
// #include <bits/stdc++.h>

using namespace std;

struct arguments{
    ArffData * train;
    ArffData *  test;
    int * predictions;
    int k;
    int id;
    int n_threads;
};

sem_t * sem_main;

float distance(ArffInstance* a, ArffInstance* b) {
    float sum = 0;
    
    for (int i = 0; i < a->size()-1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    }
    
    return sum;
}

void* KNN(void* data) {
    // Implements a parallel kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.
    struct arguments * args = (struct arguments *)data;
    ArffData * train = args->train;
    ArffData * test = args->test;
    int * predictions = args->predictions;
    int k = args->k;
    int id = args->id;
    int n_threads = args->n_threads;

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*)calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++) {candidates[i] = FLT_MAX;}
    int num_classes = train->num_classes();
    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*)calloc(num_classes, sizeof(int));
    int start = id * test->num_instances()/ n_threads;
    int end = min(test->num_instances(), (id + 1) * test->num_instances()/ n_threads);
    for (int queryIndex = start; queryIndex < end; queryIndex++) {
        for (int keyIndex = 0; keyIndex < train->num_instances(); keyIndex++) {
            float dist = distance(test->get_instance(queryIndex), train->get_instance(keyIndex));
            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(dist < candidates[2*c]){
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2*c] = dist;
                    // class value
                    candidates[2*c+1] = train->get_instance(keyIndex)->get(train->num_attributes() - 1)->operator float();
                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k;i++) {
            classCounts[(int)candidates[2*i+1]] += 1;
        }
        
        int max = -1;
        int max_index = 0;
        for(int i = 0; i < num_classes;i++) {
            if(classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;
        for(int i = 0; i < 2*k; i++) {candidates[i] = FLT_MAX;}
        memset(classCounts, 0, num_classes * sizeof(int));
    }
    sem_post(sem_main);
    pthread_exit(0);
}

int * computeConfusionMatrix(int * predictions, ArffData * dataset) {
    // matrix size numberClasses x numberClasses
    int * confusionMatrix = (int *)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int));
    
    for (int i = 0; i < dataset->num_instances(); i++) {
        // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int * confusionMatrix, ArffData * dataset) {
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++) {
        // elements in the diagonal are correct predictions
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i];
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: ./main datasets/train.arff datasets/test.arff k n_threads" << endl;
        exit(0);
    }

    int n_threads = strtol(argv[4], NULL, 10);
    pthread_t *threads;
    threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));
    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData * train = parserTrain.parse();
    ArffData * test = parserTest.parse();
    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int * predictions = (int*)malloc(test->num_instances() * sizeof(int));
    struct timespec start, end;
    struct arguments arg_array[n_threads];
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //  linux can use sem_init, mac can't
    // sem_init(&sem_main, 0, n_threads);
    sem_open("/sem_main", NULL, NULL, n_threads);
    for (int i = 0; i < n_threads; i++) {
        arg_array[i].train = train;
        arg_array[i].test = test;
        arg_array[i].predictions = predictions;
        arg_array[i].k = k;
        arg_array[i].id = i;
        arg_array[i].n_threads = n_threads;
        pthread_create(&threads[i], NULL, KNN, (void*) &arg_array[i]);
    }
    sem_wait(sem_main);
    for (int i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    free(threads);
    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier using %i threads for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n",
                        k, n_threads, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);
}
