#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "config.hpp"
#include "mpi.h"

using namespace std;

long double F(long double x, long double y) {
  return sqrt(x*x + y*y);
}

void set_time(int rank) {
  srand(rank);
}

Point get_point() {
  long double x = ((long double)rand()) / RAND_MAX * (b_x - a_x) + a_x;
  long double y = ((long double)rand()) / RAND_MAX * (b_y - a_y) + a_y;
  long double z = ((long double)rand()) / RAND_MAX * (b_z - a_z) + a_z;

  Point res;
  res.x = x;
  res.y = y;
  res.z = z;
  return res;
}

long double step(long double prev_s) {
  Point curr = get_point();
  bool is_in_circle = curr.x * curr.x + curr.y * curr.y - curr.z * curr.z < EPS_CALC;
  // z is automatically from 0 to 1
  if (is_in_circle) {
    prev_s += F(curr.x, curr.y);
  }
  return prev_s;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    cerr << "Invalid count of arguments";
  } else {
    long double target_eps;
    stringstream convert(argv[1]);
    if (!(convert >> target_eps))
		  target_eps = DEFAULT_EPS;
    cerr << target_eps << endl;
    long double v = (b_x - a_x) * (b_y - a_y) * (b_z - a_z);
    long double s;
    long double s_per_process = 0;
    int n = 0;

    //process init
    int size, rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long double start_time = MPI_Wtime();
    long double max_time;
    set_time(rank);

    while (s_per_process == 0 || abs(v * s / n - VALUE_TARGET) > target_eps) {
      long double partition_s = 0;
      for (int i = 0; i < PARTITION_CNT; ++i) {
        partition_s = step(partition_s);
      }
      n += PARTITION_CNT * size;
      s_per_process += partition_s;
     
      MPI_Reduce(&s_per_process, &s, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Bcast(&s, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

      cerr << rank << ' ' << abs(v * s / n - VALUE_TARGET) << endl; 
      MPI_Barrier(MPI_COMM_WORLD);
    }
    long double curr_process_time = MPI_Wtime() - start_time;
    MPI_Reduce(&curr_process_time, &max_time, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    long double integral = v * s / n;

    if (rank == 0) {
      cout << integral << ' ' << abs(VALUE_TARGET - integral) << ' ';
      cout << n << ' ' << max_time << endl;
    }
  }
  
  return 0;
}
