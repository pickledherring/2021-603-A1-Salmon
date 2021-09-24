## Serial, multithreaded, and MPI implementations of a KNN
---
`main` is the serial version, `main_par` is the pthread version, and `main_mpi` is the mpi version.

Run the code with `./<version> datasets/<size>-train.arff datasets/<size>-test.arff <K> <#threads>`.\
Leave off `#threads` if running `main`.

For example:\
`./main_par datasets/small-train.arff datasets/small-test.arff 3 8`

For MPI, run `mpiexec [-np <#processes>] [--oversubscribe] ./main_mpi datasets/<size>-train.arff datasets/<size>-test.arff <K>`

For example:\
`mpiexec -np 4 --oversubscribe ./main_mpi datasets/small-train.arff datasets/small-test.arff 3`