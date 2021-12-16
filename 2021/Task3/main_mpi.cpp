#include <cmath>
#include <iostream>
#include <time.h>
#include <vector>

#include "config.hpp"
#include "mpi.h"

using namespace std;

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
    vector< vector<double> > &u,
    vector< vector<double> > &v,
    double h1, double h2,
    int i_st, int i_end,
    int j_st, int j_end
)
{
    double sum1 = 0;
    for (int i = i_st; i < i_end; ++i) {
        double sum2 = 0;
        for (int j = j_st; j < j_end; ++j) {
            double ro_1 = (i == 0 || i == M) ? 0.5 : 1;
            double ro_2 = (j == 0 || j == N) ? 0.5 : 1;
            sum2 += h2 * ro_1 * ro_2 * u[i][j] * v[i][j];
        }
        sum1 += h1 * sum2;
    }
    return sum1;
}

double norm(
    vector< vector<double> > &u,
    double h1, double h2,
    int i_st, int i_end,
    int j_st, int j_end
)
{
    return sqrt(scalar(u, u, h1, h2, i_st, i_end, j_st, j_end));
}

double wx(int i, int j, vector< vector<double> > &w)
{
    return (w[i + 1][j] - w[i][j]) / h1;
}

double wx_(int i, int j, vector< vector<double> > &w)
{
    return wx(i - 1, j, w);
}

double wy(int i, int j, vector< vector<double> > &w)
{
    return (w[i][j + 1] - w[i][j]) / h2;
}

double wy_(int i, int j, vector< vector<double> > &w)
{
    return wy(i, j - 1, w);
}

double wx_x(int i, int j, vector< vector<double> > &w) 
{
    return (1 / h1) * (wx(i, j, w) - wx_(i, j, w));
}

double wy_y(int i, int j, vector< vector<double> > &w)
{
    return (1 / h2) * (wy(i, j, w) - wy_(i, j, w));
}

double delta_w(int i, int j, vector< vector<double> > &w)
{
    return wx_x(i, j, w) + wy_y(i, j, w);
}

double A(int i, int j, vector< vector<double> > &w, vector<double> &x, vector<double> &y)
{
    if ((1 <= i && i <= M - 1) && (1 <= j && j <= N - 1)) {
        return -delta_w(i, j, w) + q(x[i], y[j]) * w[i][j];
    } else if (i == M && (1 <= j && j <= N - 1)){
        return (2 / h1) * wx_(M, j, w) + (q(x[M], y[j]) + 2 / h1) * w[M][j] - wy_y(M, j, w);
    } else if (i == 0 && (1 <= j && j <= N - 1)) {
        return (-2 / h1) * wx_(1, j, w) + (q(x[0], y[j]) + 2 / h1) * w[0][j] - wy_y(0, j, w);
    } else if (j == N && (1 <= i && i <= M - 1)) {
        return (2 / h2) * wy_(i, N, w) + (q(x[i], y[N]) + 2 / h2) * w[i][N] - wx_x(i, N, w);
    } else if (j == 0 && (1 <= i && i <= M - 1)) {
        return (-2 / h2) * wy_(i, 1, w) + (q(x[i], y[0]) + 2 / h2) * w[i][0] - wx_x(i, 0, w);
    } else if (i == 0 && j == 0) {
        double fp = (-2 / h1) * wx_(1, 0, w) + (-2 / h2) * wy_(0, 1, w);
        double sp = (q(x[0], y[0]) + 2 / h1 + 2 / h2) * w[0][0];
        return fp + sp;
    } else if (i == M && j == 0) {
        double fp = (2 / h1) * wx_(M, 0, w) + (-2 / h2) * wy_(M, 1, w);
        double sp = (q(x[M], y[0]) + 2 / h1 + 2 / h2) * w[M][0];
        return fp + sp;
    } else if (i == M && j == N) {
        double fp = (2 / h1) * wx_(M, N, w) + (2 / h2) * wy_(M, N, w);
        double sp = (q(x[M], y[N]) + 2 / h1 + 2 / h2) * w[M][N];
        return fp + sp;
    } else if (i == 0 && j == N) {
        double fp = (-2 / h1) * wx_(1, N, w) + (2 / h2) * wy_(0, N, w);
        double sp = (q(x[0], y[N]) + 2 / h1 + 2 / h2) * w[0][N];
        return fp + sp;
    }
    cerr << "Wrong args for A(i, j, w, x, y)";
    return -1;
}

void init_B(vector< vector<double> > &B, vector<double> &x, vector<double> &y)
{
    for (int i = 0; i < B.size(); ++i) {
        for (int j = 0; j < B[i].size(); ++j) {
            if ((1 <= i && i <= M - 1) && (1 <= j && j <= N - 1)) {
                B[i][j] = F(x[i], y[j]);
            } else if (i == M && (1 <= j && j <= N - 1)){
                B[i][j] = F(x[M], y[j]) + (2 / h1) * psi(x[M], y[j]);
            } else if (i == 0 && (1 <= j && j <= N - 1)) {
                B[i][j] = F(x[0], y[j]) + (2 / h1) * psi(x[0], y[j]);
            } else if (j == N && (1 <= i && i <= M - 1)) {
                B[i][j] = F(x[i], y[N]) + (2 / h2) * psi(x[i], y[N]);
            } else if (j == 0 && (1 <= i && i <= M - 1)) {
                B[i][j] = F(x[i], y[0]) + (2 / h2) * psi(x[i], y[0]);
            } else if (i == 0 && j == 0) {
                double psi00 = (psi(x[1], y[0]) + psi(x[0], y[1])) / 2;
                B[i][j] = F(x[i], y[j]) + (2 / h1 + 2 / h2) * psi00;
            } else if (i == 0 && j == N) {
                double psi0N = (psi(x[1], y[N]) + psi(x[0], y[N - 1])) / 2;
                B[i][j] = F(x[i], y[j]) + (2 / h1 + 2 / h2) * psi0N;
            } else if (i == M && j == N) {
                double psiMN = (psi(x[M - 1], y[N]) + psi(x[M], y[N - 1])) / 2;
                B[i][j] = F(x[i], y[j]) + (2 / h1 + 2 / h2) * psiMN;
            } else if (i == M && j == 0) {
                double psiM0 = (psi(x[M - 1], y[0]) + psi(x[M], y[1])) / 2;
                B[i][j] = F(x[i], y[j]) + (2 / h1 + 2 / h2) * psiM0;
            }
        }
    }
    return;
}

double check_eps(
    vector< vector<double> > &target_w,
    vector< vector<double> > &curr_w,
    int i_st, int i_end,
    int j_st, int j_end
)
{
    vector< vector<double> > tmp_w(M + 1, vector<double> (N + 1));
    for (int i = i_st; i < i_end; ++i) {
        for (int j = j_st; j < j_end; ++j) {
            tmp_w[i][j] = target_w[i][j] - curr_w[i][j];
        }
    }
    return norm(tmp_w, h1, h2, i_st, i_end, j_st, j_end);
}

void get_cnt_per_domain(int &cnt_x_per_domain, int &cnt_y_per_domain)
{
    for (int k = 1; k * k <= N; ++k) { // Not ultimate solution, only in case rectangle domain
        if (N % k) {
            continue;
        }
        int l = (k * 3) / 2 + 1;
        for(; l <= k * 2; ++l) {
            if (!(M % l)) {
                cnt_y_per_domain = k;
                cnt_x_per_domain = l;
            }
        }
    }
    return;
}

int main()
{   
    vector<double> x(M + 1);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = A1 + i * h1;
    }
    
    vector<double> y(N + 1);
    for (int j = 0; j < y.size(); ++j) {
        y[j] = B1 + j * h2;
    }
    
    vector< vector<double> > B(M + 1, vector<double> (N + 1, 0));
    init_B(B, x, y);
    
    vector< vector<double> > target_w(M + 1, vector<double> (N + 1));
    for (int i = 0; i < target_w.size(); ++i) {
        for (int j = 0; j < target_w[i].size(); ++j) {
            target_w[i][j] = u(x[i], y[j]);
        }
    }

    int cnt_x_per_domain = -1; //minirectangles that divide M
    int cnt_y_per_domain = -1; //minirectangles that divide N
    get_cnt_per_domain(cnt_x_per_domain, cnt_y_per_domain);
    if (cnt_x_per_domain == -1 || cnt_y_per_domain == -1) {
        cerr << "Could not create domains" << endl;
        return -1;
    }
    int cnt_x_domains = M / cnt_x_per_domain;
    int cnt_y_domains = N / cnt_y_per_domain;
    //process init
    int size, rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double start_time = MPI_Wtime();
    double max_time;
    int mx_iterations;
    int mx_curr_iterations = 0;

    vector< vector<double> > computed_w(M + 1, vector<double> (N + 1));
    vector< vector<double> > curr_w(M + 1, vector<double> (N + 1, 0));

    for (int i = 0; i < (cnt_x_domains / size + ((cnt_x_domains % size) > 0)); ++i) {
        int curr_st_d = i * size + rank;
        int curr_i_st = curr_st_d * cnt_x_per_domain;
        if (curr_i_st > M) {
            break;
        }
        int curr_i_end = curr_i_st + cnt_x_per_domain;

        double target_norm = check_eps(target_w, curr_w, curr_i_st, curr_i_end, 0, N + 1);
        int iterations = 0;

        while (target_norm > TARGET_EPS) {
            cout << target_norm << endl;
            vector< vector<double> > rk(M + 1, vector<double> (N + 1, 0));
            for (int i = curr_i_st; i < curr_i_end; ++i) {
                for (int j = 0; j < rk[i].size(); ++j) {
                    rk[i][j] = A(i, j, curr_w, x, y) - B[i][j];
                }
            }

            vector< vector<double> > A_rk(M + 1, vector<double> (N + 1, 0));
            for (int i = curr_i_st; i < curr_i_end; ++i) {
                for (int j = 0; j < A_rk[i].size(); ++j) {
                    A_rk[i][j] = A(i, j, rk, x, y);
                }
            }

            double tmp_scalar = scalar(A_rk, rk, h1, h2, curr_i_st, curr_i_end, 0, N + 1);
            double tmp_norm = norm(A_rk, h1, h2, curr_i_st, curr_i_end, 0, N + 1);
            double tau_k_next = tmp_scalar / (tmp_norm * tmp_norm);

            for (int i = curr_i_st; i < curr_i_end; ++i) {
                for (int j = 0; j < rk[i].size(); ++j) {
                    A_rk[i][j] = curr_w[i][j]; //use A_rk as previous w
                    curr_w[i][j] = curr_w[i][j] - tau_k_next * rk[i][j];
                }
            }

            target_norm = check_eps(curr_w, A_rk, 0, M + 1, 0, N + 1);
            ++iterations;
        }
        mx_curr_iterations = std::max(iterations, mx_curr_iterations);
    }
    MPI_Reduce(&mx_curr_iterations, &mx_iterations, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    long double curr_process_time = MPI_Wtime() - start_time;
    MPI_Reduce(&curr_process_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < curr_w.size(); ++i) {
        for (int j = 0; j < curr_w[i].size(); ++j) {
            MPI_Reduce(&curr_w[i][j], &computed_w[i][j], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    if (rank == 0) {
        for (int i = 0; i < curr_w.size(); ++i) {
            for (int j = 0; j < curr_w[i].size(); ++j) {
                cerr << std::abs(target_w[i][j] - curr_w[i][j]) << ' ';
            }
            cerr << endl;
        }
        
        cout << size << ' ' << max_time << endl;
    }
    return 0;
}
