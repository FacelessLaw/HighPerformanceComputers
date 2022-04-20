#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__host__ __device__ double u(double x, double y)
{
    return sqrt(4 + x * y);
}

__host__ __device__ double k(double x, double y)
{
    return 1;
}

__host__ __device__ double q(double x, double y)
{
    if (x + y > 0)
        return x + y;
    return 0;
}

__host__ __device__ double delta_u(double x, double y)
{
    double a = x * x + y * y;
    double b = (4 + x * y) * (4 + x * y) * (4 + x * y);
    return (-0.25) * a / sqrt(b);
}
__host__ __device__ double F(double x, double y)
{
    return -delta_u(x, y) + q(x, y) * u(x, y);
}

//A1 0 A2 4  B1 0 B2 3  EPS 1e-6
__host__ __device__ double psi(double x, double y)
{
    if ((std::abs(x - 0) < 1e-6) && (0 + 1e-6 < y && y < 3 - 1e-6)) {
        return ((-y) / 4) + 2;
    } else if ((std::abs(x - 4) < 1e-6) && (0 + 1e-6 < y && y < 3 - 1e-6)) {
        return y / (4 * sqrt(1 + y)) + 2 * sqrt(1 + y);
    } else if ((0 + 1e-6 < x && x < 4 - 1e-6) && (std::abs(y - 0) < 1e-6)) {
        return ((-x) / 4) + 2;
    } else if ((0 + 1e-6 < x && x < 4 - 1e-6) && (std::abs(y - 3) < 1e-6)) {
        return x / (2 * sqrt(4 + 3 * x)) + sqrt(4 + 3 * x);
    }
    
    return u(x, y);
}

__host__ __device__ double wx(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b,
    double *w, double h1, int m, int n
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

__host__ __device__ double wx_(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b,
    double *w, double h1, int m, int n
)
{
    return wx(i - 1, j, i_st, i_end, j_st, j_end, l_b, r_b, w, h1, m, n);
}

__host__ __device__ double wy(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *b_b, double *t_b,
    double *w, double h2, int m, int n
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

__host__ __device__ double wy_(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *b_b, double *t_b,
    double *w, double h2, int m, int n
)
{
    return wy(i, j - 1, i_st, i_end, j_st, j_end, b_b, t_b, w, h2, m, n);
}

__host__ __device__ double wx_x(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b,
    double *w, double h1, int m, int n
)
{
    return (1 / h1) * (wx(i, j, i_st, i_end, j_st, j_end, l_b, r_b, w, h1, m, n) -
            wx_(i, j, i_st, i_end, j_st, j_end, l_b, r_b, w, h1, m, n));
}

__host__ __device__  double wy_y(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *b_b, double *t_b,
    double *w, double h2, int m, int n
)
{
    return (1 / h2) * (wy(i, j, i_st, i_end, j_st, j_end, b_b, t_b, w, h2, m, n) -
            wy_(i, j, i_st, i_end, j_st, j_end, b_b, t_b, w, h2, m, n));
}

__host__ __device__ double delta_w(
    int i, int j, int i_st, int i_end, int j_st, int j_end,
    double *l_b, double *r_b, double *b_b, double *t_b,
    double *w, double h1, double h2, int m, int n
)
{
    return wx_x(i, j, i_st, i_end, j_st, j_end, l_b, r_b, w, h1, m, n) +
           wy_y(i, j, i_st, i_end, j_st, j_end, b_b, t_b, w, h2, m, n);
}

__host__ __device__ double A(
    int i, int j,
    int curr_i_st, int curr_i_end,
    int curr_j_st, int curr_j_end,
    double *w, double *x, double *y,
    double *l_b, double *r_b, double *b_b, double *t_b,
    double h1, double h2,
    int m, int n
)
{
    if (
        (1 <= i + curr_i_st && i + curr_i_st <= m - 1 - 1) &&
        (1 <= j + curr_j_st && j + curr_j_st <= n - 1 - 1)
    ) {
        return -delta_w(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, b_b, t_b, w, h1, h2, m, n) +
                q(x[i + curr_i_st], y[j + curr_j_st]) * w[i * (curr_j_end - curr_j_st) + j];
    } else if (i + curr_i_st == m - 1 && (1 <= j + curr_i_st && j + curr_j_st <= n - 1 - 1)) {
        return (2 / h1) * wx_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n) + 
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1) * w[i * (curr_j_end - curr_j_st) + j] -
        wy_y(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n);
    } else if (i + curr_i_st == 0 && (1 <= j + curr_j_st && j + curr_j_st <= n - 1 - 1)) {
        return (-2 / h1) *
        wx_(i + 1, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n) +
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1) * w[i * (curr_j_end - curr_j_st) + j] -
        wy_y(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n);
    } else if (j + curr_j_st == n - 1 && (1 <= i + curr_i_st && i + curr_i_st <= m - 1 - 1)) {
        return (2 / h2) *
        wy_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n) +
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h2) * w[i * (curr_j_end - curr_j_st) + j] -
        wx_x(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n);
    } else if (j + curr_j_st == 0 && (1 <= i + curr_i_st && i + curr_i_st <= m - 1 - 1)) {
        return (-2 / h2) *
        wy_(i, j + 1, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n) +
        (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h2) * w[i * (curr_j_end - curr_j_st) + j] -
        wx_x(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n);
    } else if (i + curr_i_st == 0 && j + curr_j_st == 0) {
        double fp = (-2 / h1) *
        wx_(i + 1, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n) +
        (-2 / h2) * wy_(i, j + 1, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    } else if (i + curr_i_st == m - 1 && j + curr_j_st == 0) {
        double fp = (2 / h1) *
        wx_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n) +
        (-2 / h2) * wy_(i, j + 1, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    } else if (i + curr_i_st == m - 1 && j + curr_j_st == n - 1) {
        double fp = (2 / h1) *
        wx_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n) +
        (2 / h2) * wy_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    } else if (i + curr_i_st == 0 && j + curr_j_st == n - 1) {
        double fp = (-2 / h1) *
        wx_(i + 1, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, l_b, r_b, w, h1, m, n) +
        (2 / h2) * wy_(i, j, curr_i_st, curr_i_end, curr_j_st, curr_j_end, b_b, t_b, w, h2, m, n);
        double sp = (q(x[i + curr_i_st], y[j + curr_j_st]) + 2 / h1 + 2 / h2) *
        w[i * (curr_j_end - curr_j_st) + j];
        return fp + sp;
    }
    return -1;
}

__host__ __device__ double get_B(int i, int j, double *x, double *y, double h1, double h2, int m, int n)
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

__global__ void matrix_fill_value(double *d_arr, int x_size, int y_size, double val)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_size * y_size <= index)
        return;
    d_arr[index] = val;
    return;
}

__global__ void matrix_get_rk(
    double *d_rk,
    int i_st, int i_end, int j_st, int j_end,
    double *d_curr_w, double *d_x, double *d_y,
    double *d_left_border, double *d_right_border, double *d_bottom_border, double *d_top_border,
    double h1, double h2,
    int m, int n
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i_end - i_st - 1) * (j_end - j_st) + (j_end - j_st - 1) < index)
        return;
    
    int i = index / (j_end - j_st);
    int j = index % (j_end - j_st);

    d_rk[index] = A(
        i, j, i_st, i_end, j_st, j_end,
        d_curr_w, d_x, d_y, d_left_border, d_right_border, d_bottom_border, d_top_border, h1, h2, m, n
    ) - get_B(i + i_st, j + j_st, d_x, d_y, h1, h2, m, n);
    
    return;
}

__global__ void matrix_get_scalars(
    double *d_s1, double *d_s2,
    int i_st, int i_end, int j_st, int j_end,
    double *d_rk, double *d_x, double *d_y,
    double *d_left_border, double *d_right_border, double *d_bottom_border, double *d_top_border,
    double h1, double h2,
    int m, int n
)
{
    // int block_size = 1024;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((i_end - i_st - 1) * (j_end - j_st) + (j_end - j_st - 1) < index) {
        return;
    }

    int i = index / (j_end - j_st);
    int j = index % (j_end - j_st);

    double ro_1 = (i + i_st == 0 || i + i_st == m - 1) ? 0.5 : 1;
    double ro_2 = (j + j_st == 0 || j + j_st == n - 1) ? 0.5 : 1;
    double A_rk = A(
        i, j, i_st, i_end, j_st, j_end,
        d_rk, d_x, d_y, d_left_border, d_right_border, d_bottom_border, d_top_border, h1, h2, m, n
    ) - get_B(i + i_st, j + j_st, d_x, d_y, h1, h2, m, n);
    d_s1[index] = ro_1 * ro_2 * A_rk * d_rk[i * (j_end - j_st) + j];
    d_s2[index] = ro_1 * ro_2 * A_rk * A_rk;
    return;
}

__global__ void matrix_get_next_w(
    int i_st, int i_end, int j_st, int j_end,
    double *d_curr_w, double *d_rk, double tau_k_next
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i_end - i_st - 1) * (j_end - j_st) + j_end - j_st - 1 < index)
        return;
    
    double rkij =  d_rk[index]; //use rk as previous w
    d_rk[index] = d_curr_w[index]; 
    d_curr_w[index] = d_curr_w[index] - tau_k_next * rkij;
    return;
}

__global__ void matrix_get_new_scalar(
    double *d_s,
    int i_st, int i_end, int j_st, int j_end,
    double *d_curr_w, double *d_rk,
    int m, int n
)
{
    // int block_size = 1024;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i_end - i_st - 1) * (j_end - j_st) + (j_end - j_st - 1) < index) {
        return;
    }

    int i = index / (j_end - j_st);
    int j = index % (j_end - j_st);

    double ro_1 = (i + i_st == 0 || i + i_st == m - 1) ? 0.5 : 1;
    double ro_2 = (j + j_st == 0 || j + j_st == n - 1) ? 0.5 : 1;
    double curr_val = d_curr_w[index] - d_rk[index];
    d_s[index] = ro_1 * ro_2 * curr_val * curr_val;
    return;
}
