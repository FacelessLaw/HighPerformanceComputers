#include <iomanip>
#include <iostream>
#include <sstream>
#include <time.h>
#include <omp.h>

#include "cuda_utils.cu"
#include "mpi.h"
#include "utils.hpp"

double scalar(
    double *u,
    double *v,
    double h1, double h2,
    int i_st, int i_end,
    int j_st, int j_end,
    int m, int n
)
{
    double sum1 = 0;
    #pragma omp parallel for reduction(+: sum1)
    for (int i = i_st; i < i_end; ++i) {
        double sum2 = 0;
        #pragma omp parallel for reduction(+: sum2)
        for (int j = j_st; j < j_end; ++j) {
            double ro_1 = (i == 0 || i == m - 1) ? 0.5 : 1;
            double ro_2 = (j == 0 || j == n - 1) ? 0.5 : 1;
            sum2 += h2 * ro_1 * ro_2 * u[i * n + j] * v[i * n + j];
        }
        sum1 += h1 * sum2;
    }
    return sum1;
}

double norm(
    double *u,
    double h1, double h2,
    int i_st, int i_end,
    int j_st, int j_end,
    int m, int n
)
{
    return sqrt(scalar(u, u, h1, h2, i_st, i_end, j_st, j_end, m, n));
}

void get_cnt_per_domain(int &cnt_x_per_domain, int &cnt_y_per_domain)
{
    bool found = false;
    for (int k = 1; k * k <= N && !found; ++k) { // Not ultimate solution, only in case rectangle domain
        if (N % k) {
            continue;
        }
        int l = (k * 3) / 2 + 1;
        for(; l <= k * 2 && !found; ++l) {
            if (!(M % l)) {
                found = true;
                cnt_y_per_domain = k;
                cnt_x_per_domain = l;
            }
        }
    }
    return;
}

void left_neighbour(
    int curr_i_st,
    int curr_i_end,
    int curr_j_st,
    int curr_j_end,
    int size,
    int &req_i,
    MPI_Request *requests,
    MPI_Status *statuses,
    double *left_border,
    double *arr,
    int l_neighbour,
    int m,
    int n
)
{
    if (curr_i_st != 0) {
        double *buf = new double[curr_j_end - curr_j_st]();
        for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
            buf[j] = arr[0 * (curr_j_end - curr_j_st) + j];
        }
        MPI_Isend(buf, curr_j_end - curr_j_st, MPI_DOUBLE, l_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
        delete [] buf;

        MPI_Irecv(left_border, curr_j_end - curr_j_st, MPI_DOUBLE, l_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
    }

    return;
}

void right_neighbour(
    int curr_i_st,
    int curr_i_end,
    int curr_j_st,
    int curr_j_end,
    int size,
    int &req_i,
    MPI_Request *requests,
    MPI_Status *statuses,
    double *right_border,
    double *arr,
    int r_neighbour,
    int m,
    int n
)
{
    if (curr_i_end != M + 1) {
        double *buf = new double[curr_j_end - curr_j_st]();
        for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
            buf[j] = arr[(curr_i_end - curr_i_st - 1) * (curr_j_end - curr_j_st) + j];
        }
        MPI_Isend(buf, curr_j_end - curr_j_st, MPI_DOUBLE, r_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
        delete [] buf;

        MPI_Irecv(right_border, curr_j_end - curr_j_st, MPI_DOUBLE, r_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
    }
    return;
}

void bottom_neighbour(
    int curr_j_st,
    int curr_j_end,
    int curr_i_st,
    int curr_i_end,
    int size,
    int &req_i,
    MPI_Request *requests,
    MPI_Status *statuses,
    double *bottom_border,
    double *arr,
    int b_neighbour,
    int m,
    int n
)
{
    if (curr_j_st != 0) {
        double *buf = new double[curr_i_end - curr_i_st]();
        for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
            buf[i] = arr[i * (curr_j_end - curr_j_st) + 0];
        }
        MPI_Isend(buf, curr_i_end - curr_i_st, MPI_DOUBLE, b_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
        delete [] buf;

        MPI_Irecv(bottom_border, curr_i_end - curr_i_st, MPI_DOUBLE, b_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
    }
    return;
}

void top_neighbour(
    int curr_j_st,
    int curr_j_end,
    int curr_i_st,
    int curr_i_end,
    int size,
    int &req_i,
    MPI_Request *requests,
    MPI_Status *statuses,
    double *top_border,
    double *arr,
    int t_neighbour,
    int m,
    int n
)
{
    if (curr_j_end != N + 1) {
        double *buf = new double[curr_i_end - curr_i_st]();
        for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
            buf[i] = arr[i * (curr_j_end - curr_j_st) + curr_j_end - curr_j_st - 1];
        }
        MPI_Isend(buf, curr_i_end - curr_i_st, MPI_DOUBLE, t_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
        delete [] buf;

        MPI_Irecv(top_border, curr_i_end - curr_i_st, MPI_DOUBLE, t_neighbour, 0, MPI_COMM_WORLD, &requests[req_i]);
        ++req_i;
    }
    return;
}

void fill_start_value(int i_st, int i_end, int j_st, int j_end, int m, int n, double val, double *arr)
{
    int block_size = 1024;
    int grid_size = ((i_end - i_st) * (j_end - j_st) - 1) / block_size + 1;

    size_t bytes = (i_end - i_st) * (j_end - j_st) * sizeof(double);

    double *d_arr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);
    
    matrix_fill_value<<<grid_size, block_size>>>(d_arr, i_end - i_st, j_end - j_st, val);
    // cudaDeviceSynchronize();

    cudaMemcpy(arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    return;
}

void get_rk(
    int i_st, int i_end, int j_st, int j_end,
    double * rk, double *curr_w, double *x, double *y,
    double *left_border, double *right_border, double *bottom_border, double *top_border,
    int m, int n
)
{
    int block_size = 1024;
    int grid_size = ((i_end - i_st) * (j_end - j_st) - 1) / block_size + 1;

    size_t bytes = (i_end - i_st) * (j_end - j_st) * sizeof(double);
    double *d_curr_w;
    double *d_rk;
    double *d_left_border;
    double *d_right_border;
    double *d_bottom_border;
    double *d_top_border;
    double *d_x;
    double *d_y;

    cudaMalloc(&d_curr_w, bytes);
    cudaMalloc(&d_rk, bytes);
    cudaMalloc(&d_left_border, (j_end - j_st) * sizeof(double));
    cudaMalloc(&d_right_border, (j_end - j_st) * sizeof(double));
    cudaMalloc(&d_bottom_border, (i_end - i_st) * sizeof(double));
    cudaMalloc(&d_top_border, (i_end - i_st) * sizeof(double));
    cudaMalloc(&d_x, (m) * sizeof(double));
    cudaMalloc(&d_y, (n) * sizeof(double));
    cudaMemcpy(d_curr_w, curr_w, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_left_border, left_border, (j_end - j_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_border, right_border, (j_end - j_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bottom_border, bottom_border, (i_end - i_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_border, top_border, (i_end - i_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, (m) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, (n) * sizeof(double), cudaMemcpyHostToDevice);
    
    matrix_get_rk<<<grid_size, block_size>>>(
        d_rk,
        i_st, i_end, j_st, j_end,
        d_curr_w, d_x, d_y, d_left_border, d_right_border, d_bottom_border, d_top_border, h1, h2, m, n
    );
    // cudaDeviceSynchronize();

    cudaMemcpy(rk, d_rk, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_curr_w);
    cudaFree(d_rk);
    cudaFree(d_left_border);
    cudaFree(d_right_border);
    cudaFree(d_bottom_border);
    cudaFree(d_top_border);
    cudaFree(d_x);
    cudaFree(d_y);

    return;
}

void get_scalars(
    double &scalar1, double &scalar2,
    int i_st, int i_end, int j_st, int j_end,
    double *rk, double *x, double *y,
    double *left_border, double *right_border, double *bottom_border, double *top_border,
    int m, int n
)
{
    int block_size = 1024;
    int grid_size = ((i_end - i_st) * (j_end - j_st) - 1) / block_size + 1;

    size_t bytes = (i_end - i_st) * (j_end - j_st) * sizeof(double);
    
    thrust::host_vector<double> sums1((i_end - i_st) * (j_end - j_st));
    thrust::device_vector<double> d_sums1((i_end - i_st) * (j_end - j_st));
    double *d_s1 = thrust::raw_pointer_cast(&d_sums1[0]);
    thrust::host_vector<double> sums2((i_end - i_st) * (j_end - j_st));
    thrust::device_vector<double> d_sums2((i_end - i_st) * (j_end - j_st));
    double *d_s2 = thrust::raw_pointer_cast(&d_sums2[0]);

    double *d_rk;
    double *d_left_border;
    double *d_right_border;
    double *d_bottom_border;
    double *d_top_border;
    double *d_x;
    double *d_y;
    cudaMalloc(&d_rk, bytes);
    cudaMalloc(&d_left_border, (j_end - j_st) * sizeof(double));
    cudaMalloc(&d_right_border, (j_end - j_st) * sizeof(double));
    cudaMalloc(&d_bottom_border, (i_end - i_st) * sizeof(double));
    cudaMalloc(&d_top_border, (i_end - i_st) * sizeof(double));
    cudaMalloc(&d_x, (m) * sizeof(double));
    cudaMalloc(&d_y, (n) * sizeof(double));

    cudaMemcpy(d_rk, rk, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_left_border, left_border, (j_end - j_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_border, right_border, (j_end - j_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bottom_border, bottom_border, (i_end - i_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_border, top_border, (i_end - i_st) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, (m) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, (n) * sizeof(double), cudaMemcpyHostToDevice);
    
    matrix_get_scalars<<<grid_size, block_size>>>(
        d_s1, d_s2,
        i_st, i_end, j_st, j_end,
        d_rk, d_x, d_y, d_left_border, d_right_border, d_bottom_border, d_top_border, h1, h2, m, n
    );
    // cudaDeviceSynchronize();

    sums1 = d_sums1;
    sums2 = d_sums2;
    
    scalar1 = thrust::reduce(sums1.begin(), sums1.end(), (double) 0, thrust::plus<double>());
    scalar2 = thrust::reduce(sums2.begin(), sums2.end(), (double) 0, thrust::plus<double>());

    cudaFree(d_rk);
    cudaFree(d_left_border);
    cudaFree(d_right_border);
    cudaFree(d_bottom_border);
    cudaFree(d_top_border);
    cudaFree(d_x);
    cudaFree(d_y);

    return;
}

void get_w_next(
    int i_st, int i_end, int j_st, int j_end,
    double *curr_w, double *rk, double tau_k_next
)
{
    int block_size = 1024;
    int grid_size = ((i_end - i_st) * (j_end - j_st) - 1) / block_size + 1;

    size_t bytes = (i_end - i_st) * (j_end - j_st) * sizeof(double);

    double *d_curr_w;
    double *d_rk;
    cudaMalloc(&d_curr_w, bytes);
    cudaMalloc(&d_rk, bytes);
    cudaMemcpy(d_curr_w, curr_w, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk, rk, bytes, cudaMemcpyHostToDevice);
    
    matrix_get_next_w<<<grid_size, block_size>>>(
        i_st, i_end, j_st, j_end,
        d_curr_w, d_rk, tau_k_next
    );
    // cudaDeviceSynchronize();

    cudaMemcpy(curr_w, d_curr_w, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(rk, d_rk, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_curr_w);
    cudaFree(d_rk);

    return;
}

void get_new_scalar(
    double &scalar,
    int i_st, int i_end, int j_st, int j_end,
    double *curr_w, double *rk,
    int m, int n
)
{
    int block_size = 1024;
    int grid_size = ((i_end - i_st) * (j_end - j_st) - 1) / block_size + 1;

    size_t bytes = (i_end - i_st) * (j_end - j_st) * sizeof(double);
    
    thrust::host_vector<double> sums((i_end - i_st) * (j_end - j_st));
    thrust::device_vector<double> d_sums((i_end - i_st) * (j_end - j_st));
    double *d_s = thrust::raw_pointer_cast(&d_sums[0]);

    double *d_curr_w;
    double *d_rk;

    cudaMalloc(&d_curr_w, bytes);
    cudaMalloc(&d_rk, bytes);

    cudaMemcpy(d_curr_w, curr_w, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk, rk, bytes, cudaMemcpyHostToDevice);
    
    matrix_get_new_scalar<<<grid_size, block_size>>>(
        d_s,
        i_st, i_end, j_st, j_end,
        d_curr_w, d_rk, m, n
    );
    // cudaDeviceSynchronize();
    
    sums = d_sums;
    scalar = thrust::reduce(sums.begin(), sums.end(), (double) 0, thrust::plus<double>());

    cudaFree(d_curr_w);
    cudaFree(d_rk);

    return;
}

int main(int argc, char **argv)
{
    double *x = new double[M + 1];
    for (size_t i = 0; i < M + 1; ++i) {
        x[i] = A1 + i * h1;
    }
    
    double *y = new double[N + 1];
    for (size_t j = 0; j < N + 1; ++j) {
        y[j] = B1 + j * h2;
    }

    //process init
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();
    double max_time;
    int mx_iterations;

    vector<int> istart(size);
    vector<int> jstart(size);
    vector<int> ilen(size);
    vector<int> jlen(size);
    int *l_neighbours = new int[size];
    int *r_neighbours = new int[size];
    int *b_neighbours = new int[size];
    int *t_neighbours = new int[size];
    
    domains_for_procs(size, istart, jstart, ilen, jlen, l_neighbours, r_neighbours, b_neighbours, t_neighbours);

    int curr_i_st = istart[rank];
    int curr_i_end = curr_i_st + ilen[rank];
    int curr_j_st = jstart[rank];
    int curr_j_end = curr_j_st + jlen[rank];
    double * curr_w = new double[(curr_i_end - curr_i_st) * (curr_j_end - curr_j_st)]();

    double *left_border_w = new double[curr_j_end - curr_j_st];
    double *right_border_w = new double[curr_j_end - curr_j_st];
    double *left_border_r = new double[curr_j_end - curr_j_st];
    double *right_border_r = new double[curr_j_end - curr_j_st];
    
    double *top_border_w = new double[curr_i_end - curr_i_st];
    double *bottom_border_w = new double[curr_i_end - curr_i_st];
    double *top_border_r = new double[curr_i_end - curr_i_st];
    double *bottom_border_r = new double[curr_i_end - curr_i_st];

    fill_start_value(curr_i_st, curr_i_end, curr_j_st, curr_j_end, M + 1, N + 1, 3, curr_w);

    int req_cnt = (curr_i_st != 0) * 2 + (curr_i_end != M + 1) * 2
                + (curr_j_st != 0) * 2 + (curr_j_end != N + 1) * 2;

    MPI_Request requests_w[req_cnt];
    MPI_Request requests_r[req_cnt];
    MPI_Status statuses_w[req_cnt];
    MPI_Status statuses_r[req_cnt];

    double target_norm = TARGET_EPS * 100;

    double curr_scalar = 0;
    double curr_norm2 = 0;
    int iterations = 0;

    while (target_norm > TARGET_EPS) {
        if (rank == 0)
            cerr << target_norm << endl;
        
        int req_i = 0;
        int m = M + 1;
        int n = N + 1;
        left_neighbour(
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            size, req_i,
            requests_w, statuses_w, left_border_w, curr_w, l_neighbours[rank],
            m, n
        );
        right_neighbour(
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            size, req_i,
            requests_w, statuses_w, right_border_w, curr_w, r_neighbours[rank],
            m, n
        );
        bottom_neighbour(
            curr_j_st, curr_j_end, curr_i_st, curr_i_end,
            size, req_i,
            requests_w, statuses_w, bottom_border_w, curr_w, b_neighbours[rank],
            m, n
        );
        top_neighbour(
            curr_j_st, curr_j_end, curr_i_st, curr_i_end,
            size, req_i,
            requests_w, statuses_w, top_border_w, curr_w, t_neighbours[rank],
            m, n
        );
        MPI_Waitall(req_cnt, requests_w, statuses_w);
        
        double * rk = new double[(curr_i_end - curr_i_st) * (curr_j_end - curr_j_st)]();
        get_rk(
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            rk, curr_w, x, y,
            left_border_w, right_border_w, bottom_border_w, top_border_w,
            m, n
        );
        
        req_i = 0;
        left_neighbour(
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            size, req_i,
            requests_r, statuses_r, left_border_r, rk, l_neighbours[rank],
            m, n
        );
        right_neighbour(
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            size, req_i,
            requests_r, statuses_r, right_border_r, rk, r_neighbours[rank],
            m, n
        );
        bottom_neighbour(
            curr_j_st, curr_j_end, curr_i_st, curr_i_end,
            size, req_i,
            requests_r, statuses_r, bottom_border_r, rk, b_neighbours[rank],
            m, n
        );
        top_neighbour(
            curr_j_st, curr_j_end, curr_i_st, curr_i_end,
            size, req_i,
            requests_r, statuses_r, top_border_r, rk, t_neighbours[rank],
            m, n
        );
        MPI_Waitall(req_cnt, requests_r, statuses_r);
        
        double scalar1 = 0;
        double scalar2 = 0;
        get_scalars(
            scalar1, scalar2,
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            rk, x, y,
            left_border_w, right_border_w, bottom_border_w, top_border_w,
            m, n
        );
        scalar1 *= h1 * h2;
        scalar2 *= h1 * h2;
        
        MPI_Allreduce(&scalar1, &curr_scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&scalar2, &curr_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double tau_k_next = curr_scalar /  curr_norm2;

        get_w_next(
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            curr_w, rk, tau_k_next
        );

        double curr_norm = 0;
        get_new_scalar(
            curr_norm,
            curr_i_st, curr_i_end, curr_j_st, curr_j_end,
            curr_w, rk,
            m, n
        );
        curr_norm = sqrt(h1 * h2 * curr_norm);

        MPI_Allreduce(&curr_norm, &target_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete [] rk;
        ++iterations;
    }

    MPI_Reduce(&iterations, &mx_iterations, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    double curr_process_time = MPI_Wtime() - start_time;
    MPI_Reduce(&curr_process_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << size << ' ' << max_time << ' ' << iterations << endl;
    }

    for (int q = 0; q < size; ++q) {
		if (rank == q) {
			stringstream filename;
			filename << "MPI" << M << "x" << N << "_" << size << "_proc" << q << ".txt";
			cout << filename.str() << endl;
			ofstream out(filename.str().c_str());
			for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
				for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
					out << setprecision(10) << curr_w[i * (curr_j_end - curr_j_st) + j];
					if (j != curr_j_end - curr_j_st - 1)
						out << ", ";
				}
				out << "\n";
			}
		}
	}
    
    delete [] left_border_w;
    delete [] right_border_w;
    delete [] left_border_r;
    delete [] right_border_r;

    delete [] top_border_w;
    delete [] bottom_border_w;
    delete [] top_border_r;
    delete [] bottom_border_r;

    delete [] curr_w;
    delete [] l_neighbours;
    delete [] r_neighbours;
    delete [] b_neighbours;
    delete [] t_neighbours;
    MPI_Finalize();

    delete [] x;
    delete [] y;

    return 0;
}
