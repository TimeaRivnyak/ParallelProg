#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <random>
#include <ctime>
#define N 3

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  int dims[2], periods[2], coords[2];
  int up, down, left, right;
  std::vector<float> a(N*N), arec(N*N), b(N*N), brec(N*N), c(N*N);
  bool square = false;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (ceil((double)sqrt(size)) == floor((double)sqrt(size))) 
    square = true;
  if (square == false){
		if (rank == 0) std::cout << "The number of processes is not a square number" << std::endl;
		MPI_Finalize();
		return 0;
	}

  MPI_Status status;
  MPI_Comm comm_2d; 
  MPI_Request reqs[4];

  dims[0] = dims[1] = sqrt(size);
  periods[0] = periods[1] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d); 
  MPI_Comm_rank(comm_2d, &rank); 
  MPI_Cart_coords(comm_2d, rank, 2, coords);

  MPI_Cart_shift(comm_2d, 0, 1, &left, &right); 
  MPI_Cart_shift(comm_2d, 1, 1, &up, &down);

  srand (time(NULL)+rank);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      a[i*N+j] = 1 + (rand() % 10);
      b[i*N+j] = 1 + (rand() % 10);
      c[i*N+j] = 0.0f;
    }
  
  double start=MPI_Wtime();
  for (int shift=0; shift<dims[0]; shift++) { 
    printf("Rank: %d, shift: %d, A: ",rank,shift);
    copy(a.begin(),a.end(),std::ostream_iterator<float>(std::cout," "));
    std::cout <<std::endl;
    printf("Rank: %d, shift: %d, B: ",rank,shift);
    copy(b.begin(),b.end(),std::ostream_iterator<float>(std::cout," "));
    std::cout <<std::endl;

    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
          c[i*N + j] += a[i*N + k] * b[k*N + j];

    MPI_Isend(a.data(), N*N, MPI_FLOAT, left, 1, comm_2d, &reqs[0]);
    MPI_Irecv(arec.data(), N*N, MPI_FLOAT, right, 1, comm_2d, &reqs[2]); 
    MPI_Isend(b.data(), N*N, MPI_FLOAT, up, 1, comm_2d, &reqs[1]);  
    MPI_Irecv(brec.data(), N*N, MPI_FLOAT, down, 1, comm_2d, &reqs[3]);
    MPI_Wait(&reqs[2], MPI_STATUS_IGNORE);		
		MPI_Wait(&reqs[3], MPI_STATUS_IGNORE);
    
    a.clear();
    b.clear();
    copy(arec.begin(), arec.end(), back_inserter(a));
    copy(brec.begin(), brec.end(), back_inserter(b));
  } 
  double stop=MPI_Wtime();
  MPI_Comm_free(&comm_2d);
  printf("Rank %d:\n",rank);
  copy(c.begin(),c.end(),std::ostream_iterator<float>(std::cout," "));
  std::cout<<"\n";
  printf("Time: %.4fs\n",stop-start);
  MPI_Finalize();
  return 0;
}
