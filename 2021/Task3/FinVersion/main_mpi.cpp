#include <iomanip>
#include <iostream>
#include <sstream>
#include <time.h>
#include <omp.h>

#include "mpi.h"
#include "utils.hpp"

double u(double x, double y)
{
    return sqrt(4 + x * y);
}

double k(double x, double y)
{
    return 1;
}

double q(double x, double y)
{
    if (x + y > 0)
        return x + y;
    return 0;
}

double delta_u(double x, double y)
{
    double a = x * x + y * y;
    double b = (4 + x * y) * (4 + x * y) * (4 + x * y);
    return (-0.25) * a / sqrt(b);
}

double F(double x, double y)
{
    return -delta_u(x, y) + q(x, y) * u(x, y);
}

double psi(double x, double y)
{
    if ((std::abs(x - A1) < EPS) && (B1 + EPS < y && y < B2 - EPS)) {
        return ((-y) / 4) + 2;
    } else if ((std::abs(x - A2) < EPS) && (B1 + EPS < y && y < B2 - EPS)) {
        return y / (4 * sqrt(1 + y)) + 2 * sqrt(1 + y);
    } else if ((A1 + EPS < x && x < A2 - EPS) && (std::abs(y - B1) < EPS)) {
        return ((-x) / 4) + 2;
    } else if ((A1 + EPS < x && x < A2 - EPS) && (std::abs(y - B2) < EPS)) {
        return x / (2 * sqrt(4 + 3 * x)) + sqrt(4 + 3 * x);
    }
    cerr << "Wrong args for psi(x, y): " << x << ' ' << y << endl;
    return u(x, y);
}

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

double wx(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b,
    double *w, int m, int n
)
{
    double f, s;
    if (i + 1 >= i_end - i_st)
        f = r_b[j];
    else
        f = w[(i + 1) * (j_end - j_st) + j];
    
    if (i < 0)
        s = l_b[j];
    else
        s = w[i * (j_end - j_st) + j];
    return (f - s) / h1;
}

double wx_(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b,
    double *w, int m, int n
)
{
    return wx(i - 1, j, i_st, i_end, j_st, j_end, l_b, r_b, w, m, n);
}

double wy(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *b_b, double *t_b,
    double *w, int m, int n
)
{
    double f, s;
    if (j + 1 >= j_end - j_st)
        f = t_b[i];
    else
        f = w[i * (j_end - j_st) + j + 1];
    
    if (j < 0)
        s = b_b[i];
    else
        s = w[i * (j_end - j_st) + j];
    return (f - s) / h2;
}

double wy_(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *b_b, double *t_b,
    double *w, int m, int n
)
{
    return wy(i, j - 1, i_st, i_end, j_st, j_end, b_b, t_b, w, m, n);
}

double wx_x(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b,
    double *w, int m, int n
)
{
    return (1 / h1) * (wx(i, j, i_st, i_end, j_st, j_end, l_b, r_b, w, m, n) -
            wx_(i, j, i_st, i_end, j_st, j_end, l_b, r_b, w, m, n));
}

double wy_y(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *b_b, double *t_b,
    double *w, int m, int n
)
{
    return (1 / h2) * (wy(i, j, i_st, i_end, j_st, j_end, b_b, t_b, w, m, n) -
            wy_(i, j, i_st, i_end, j_st, j_end, b_b, t_b, w, m, n));
}

double delta_w(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b, double *b_b, double *t_b,
    double *w, int m, int n
)
{
    return wx_x(i, j, i_st, i_end, j_st, j_end, l_b, r_b, w, m, n) +
           wy_y(i, j, i_st, i_end, j_st, j_end, b_b, t_b, w, m, n);
}

double A(
    int i, int j,
    int curr_i_st, int curr_i_end,
    int curr_j_st, int curr_j_end,
    double *w, double *x, double *y,
    double *l_b, double *r_b, double *b_b, double *t_b,
    int m, int n
)
{
    if (
        (1 <= i + curr_i_st && i + curr_i_st <= m - 1 - 1) &&
        (1 <= j + curr_j_st && j + curr_j_st <= n - 1 - 1)
    ) {
        return -delta_w(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, b_b, t_b, w, m, n) +
                q(x[i + curr_i_st], y[j + curr_j_st]) * w[i * (curr_j_end - curr_j_st) + j];
    } else if (i + curr_i_st == m - 1 && (1 <= j + curr_i_st && j + curr_j_st <= n - 1 - 1)) {
        return (2 / h1) * wx_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n) + 
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1) * w[i * (curr_j_end - curr_j_st) + j] -
        wy_y(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n);
    } else if (i + curr_i_st == 0 && (1 <= j + curr_j_st && j + curr_j_st <= n - 1 - 1)) {
        return (-2 / h1) *
        wx_(i + 1, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n) +
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1) * w[i * (curr_j_end - curr_j_st) + j] -
        wy_y(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n);
    } else if (j + curr_j_st == n - 1 && (1 <= i + curr_i_st && i + curr_i_st <= m - 1 - 1)) {
        return (2 / h2) *
        wy_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n) +
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h2) * w[i * (curr_j_end - curr_j_st) + j] -
        wx_x(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n);
    } else if (j + curr_j_st == 0 && (1 <= i + curr_i_st && i + curr_i_st <= m - 1 - 1)) {
        return (-2 / h2) *
        wy_(i, j + 1, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n) +
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h2) * w[i * (curr_j_end - curr_j_st) + j] -
        wx_x(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n);
    } else if (i + curr_i_st == 0 && j + curr_j_st == 0) {
        double fp = (-2 / h1) *
        wx_(i + 1, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n) +
        (-2 / h2) * wy_(i, j + 1, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    } else if (i + curr_i_st == m - 1 && j + curr_j_st == 0) {
        double fp = (2 / h1) *
        wx_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n) +
        (-2 / h2) * wy_(i, j + 1, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    } else if (i + curr_i_st == m - 1 && j + curr_j_st == n - 1) {
        double fp = (2 / h1) *
        wx_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n) +
        (2 / h2) * wy_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    } else if (i + curr_i_st == 0 && j + curr_j_st == n - 1) {
        double fp = (-2 / h1) *
        wx_(i + 1, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, m, n) +
        (2 / h2) * wy_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    }
    cerr << "Wrong args for A(i, j, w, x, y)";
    return -1;
}

double get_B(int i, int j, double *x, double *y, int m, int n)
{
    if ((1 <= i && i <= m - 1 - 1) && (1 <= j && j <= n - 1 - 1)) {
        return F(x[i], y[j]);
    } else if (i == m - 1 && (1 <= j && j <= n - 1 - 1)){
        return F(x[m - 1], y[j]) + (2 / h1) * psi(x[m - 1], y[j]);
    } else if (i == 0 && (1 <= j && j <= n - 1 - 1)) {
        return F(x[0], y[j]) + (2 / h1) * psi(x[0], y[j]);
    } else if (j == n - 1 && (1 <= i && i <= m - 1 - 1)) {
        return F(x[i], y[n - 1]) + (2 / h2) * psi(x[i], y[n - 1]);
    } else if (j == 0 && (1 <= i && i <= m - 1 - 1)) {
        return F(x[i], y[0]) + (2 / h2) * psi(x[i], y[0]);
    } else if (i == 0 && j == 0) {
        double psi00 = (psi(x[1], y[0]) + psi(x[0], y[1])) / 2;
        return F(x[i], y[j]) + (2 / h1 + 2 / h2) * psi00;
    } else if (i == 0 && j == n - 1) {
        double psi0N = (psi(x[1], y[n - 1]) + psi(x[0], y[n - 1 - 1])) / 2;
        return F(x[i], y[j]) + (2 / h1 + 2 / h2) * psi0N;
    } else if (i == m - 1 && j == n - 1) {
        double psiMN = (psi(x[m - 1 - 1], y[n - 1]) + psi(x[m - 1], y[n - 1 - 1])) / 2;
        return F(x[i], y[j]) + (2 / h1 + 2 / h2) * psiMN;
    } else if (i == m - 1 && j == 0) {
        double psiM0 = (psi(x[m - 1 - 1], y[0]) + psi(x[m - 1], y[1])) / 2;
        return F(x[i], y[j]) + (2 / h1 + 2 / h2) * psiM0;
    }
    return 0;
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
    for (int i = 0; i < i_end - i_st; ++i) {
        for (int j = 0; j < j_end - j_st; ++j) {
            arr[i * (j_end - j_st) + j] = val;
        }
    }
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
        #pragma omp parallel for
        for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
            for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
                rk[i * (curr_j_end - curr_j_st) + j] = 
                A(
                    i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end,
                    curr_w, x, y, left_border_w, right_border_w, bottom_border_w, top_border_w, m, n
                ) - get_B(i + curr_i_st, j + curr_j_st, x, y, m, n);
            }
        }
        
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
        #pragma omp parallel for reduction(+: scalar1, scalar2)
        for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
            double sum1 = 0;
            double sum2 = 0;
            for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
                double ro_1 = (i + curr_i_st == 0 || i + curr_i_st == m - 1) ? 0.5 : 1;
                double ro_2 = (j + curr_j_st == 0 || j + curr_j_st == n - 1) ? 0.5 : 1;
                double A_rk = A(
                    i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end,
                    rk, x, y, left_border_r, right_border_r, bottom_border_r, top_border_r, m, n
                );
                sum1 += h2 * ro_1 * ro_2 * A_rk * rk[i * (curr_j_end - curr_j_st) + j];
                sum2 += h2 * ro_1 * ro_2 * A_rk * A_rk;
            }
            scalar1 += h1 * sum1;
            scalar2 += h1 * sum2;
        }
        
        MPI_Allreduce(&scalar1, &curr_scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&scalar2, &curr_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double tau_k_next = curr_scalar /  curr_norm2;

        #pragma omp parallel for
        for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
            for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
                double rkij =  rk[i * (curr_j_end - curr_j_st) + j];
                //use rk as previous w
                rk[i * (curr_j_end - curr_j_st) + j] = curr_w[i * (curr_j_end - curr_j_st) + j]; 
                curr_w[i * (curr_j_end - curr_j_st) + j] = curr_w[i * (curr_j_end - curr_j_st) + j] -
                                                           tau_k_next * rkij;
            }
        }

        double curr_norm = 0;
        #pragma omp parallel for reduction(+: curr_norm)
        for (int i = 0; i < curr_i_end - curr_i_st; ++i) {
            double sum = 0;
            for (int j = 0; j < curr_j_end - curr_j_st; ++j) {
                double ro_1 = (i + curr_i_st == 0 || i + curr_i_st == m - 1) ? 0.5 : 1;
                double ro_2 = (j + curr_j_st == 0 || j + curr_j_st == n - 1) ? 0.5 : 1;
                double curr_val = curr_w[i * (curr_j_end - curr_j_st) + j] - rk[i * (curr_j_end - curr_j_st) + j];
                sum += h2 * ro_1 * ro_2 * curr_val * curr_val;
            }
            curr_norm += h1 * sum;
        }
        curr_norm = sqrt(curr_norm);

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
