__author__ = 'HyNguyen'

import numpy as np
import time
from mpi4py import MPI
from vector.wordvectors import WordVectors


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    size_sample = 5000
    data_scatters = []
    start_total = 0
    if rank == 0:
        start_total = time.time()
        w2v = WordVectors.load("vector/100")
        print("Finished read wordvectors ...")

        with open("data/mini_corpus.txt", mode="r") as f:
            lines = f.readlines()
            for process_id in range(size):
                if process_id*size_sample+size_sample < len(lines):
                    data_scatters.append(lines[process_id*size_sample:process_id*size_sample+size_sample])
                else:
                    data_scatters.append(lines[process_id*size_sample:])
    else:
        w2v = None
        data_scatter = None

    w2v = comm.bcast(w2v, root = 0)
    print("Process:", rank, "broadcasted wordvectors ...")
    data_scatter = comm.scatter(data_scatters,root=0)
    print("Process:", rank, "Data scatter length: ", len(data_scatter))

    # print("Process:", rank, "data scatter[0]", data_scatter[0])
    data_matrix = []
    for i, line in enumerate(data_scatter):
        if i % 1000 == 0:
            print("Process:", rank, "preprocessed line", i)
        sentence_matrix = w2v.cae_prepare_data(line, min_length=10, max_length=100)
        if sentence_matrix is not None:
            data_matrix.append(sentence_matrix)
    data_matrix = np.array(data_matrix)
    print("Process:", rank, "Finished preprocess Scatter data...")

    data_matrix_gather = comm.gather(data_matrix, root=0)

    if rank == 0:
        data_matrix_final = data_matrix_gather[0]
        for i in range(1,len(data_matrix_gather)):
            data_matrix_final = np.concatenate((data_matrix_final,data_matrix_gather[i]))
        print("Process:", rank, "data_matrix_final.shape: ", data_matrix_final.shape)
        end_total = time.time()
        print("Process:", rank, "Total time: ", end_total - start_total, "s")
        np.save("data/mini.corpus", data_matrix_final)
        print("Process:", rank, "Save to data/mini.corpus ")