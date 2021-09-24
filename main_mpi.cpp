#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include "mpi.h"
// #include <bits/stdc++.h>
#include "bits_file/stdc++.h"

using namespace std;

struct arguments{
    ArffData * train;
    ArffData *  test;
    int * predictions;
    int k;
};

float distance(float *a, ArffInstance* b) {
    float sum = 0;
    for (int i = 0; i < b->size()-1; i++) {
        float diff = (a[i] - b->get(i)->operator float());
        sum += diff*diff;
    }
    return sum;
}

void KNN(struct arguments data) {
    /* Implements a sequential kNN where for each candidate query,
        an in-place priority queue is maintained to identify the kNN's. */
    ArffData* train = data.train;
    ArffData* test = data.test;
    int* predictions = data.predictions;
    int k = data.k;

    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    
    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*)calloc(k * 2, sizeof(float));
    for(int i = 0; i < 2 * k; i++) {candidates[i] = FLT_MAX;}

    int num_classes = train->num_classes();
    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*)calloc(num_classes, sizeof(int));
    // variables for scattering
    // getting the number of dimensions (last element is the class)
    int dim = test->get_instance(0)->size() - 1;
    float test_floats[test->num_instances()][dim];
    // I assume MPI_Scatter() can't deal with arff data, so I'm casting it to floats
    for(int queryIndex = 0; queryIndex < test->num_instances(); queryIndex++) {
        for (int i = 0; i < dim; i++) {
            test_floats[queryIndex][i] = test->get_instance(queryIndex)->get(i)->operator float();
        }
    }
    int count = test->num_instances() / ntasks;
    float rec_buffer[count][dim];
    int source = 0;
    MPI_Scatter(test_floats, count * dim, MPI_FLOAT, rec_buffer, count * dim, MPI_FLOAT, source, MPI_COMM_WORLD);

    // this will be used in MPI_Gather()
    int* gat_send = (int*)malloc(count * sizeof(int));
    for(int queryIndex = 0; queryIndex < count; queryIndex++) {
        for(int keyIndex = 0; keyIndex < train->num_instances(); keyIndex++) {
            float dist = distance(rec_buffer[queryIndex], train->get_instance(keyIndex));
            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(dist < candidates[2 * c]){
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }
                    // Set key vector as potential k NN
                    candidates[2 * c] = dist;
                    candidates[2 * c + 1] = train->get_instance(keyIndex)->get(train->num_attributes() - 1)->operator float(); // class value;
                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k; i++){
            classCounts[(int)candidates[2 * i + 1]] += 1;
        }
        
        int max = -1;
        int max_index = 0;
        for(int i = 0; i < num_classes; i++){
            if(classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }
        gat_send[queryIndex] = max_index;
        for(int i = 0; i < 2*k; i++) {candidates[i] = FLT_MAX;}
        memset(classCounts, 0, num_classes * sizeof(int));
    }
    MPI_Gather(gat_send, count, MPI_FLOAT, predictions, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset) {
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]){

    if(argc != 4)
    {
        cout << "Usage: mpiexec -np <n> ./main_mpi datasets/train.arff datasets/test.arff k" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData * train = parserTrain.parse();
    ArffData * test = parserTest.parse();
    int * predictions = (int*)malloc(test->num_instances() * sizeof(int));
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    struct arguments arg_struct;
    arg_struct.train = train;
    arg_struct.test = test;
    arg_struct.predictions = predictions;
    arg_struct.k = k;
    MPI_Init(&argc, &argv);
    KNN(arg_struct);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Finalize();
    if (rank == 0) {
        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n",
                    k, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);
    }
}
