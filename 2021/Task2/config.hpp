#include <cmath>

struct Point{
  long double x, y, z;
};

const int PARTITION_CNT = 1e4;
const long double EPS_CALC = 1e-7;
const long double DEFAULT_EPS = 1e-4;
const long double VALUE_TARGET = M_PI / 6;
const long double a_x = -1;
const long double b_x = 1;
const long double a_y = -1;
const long double b_y = 1;
const long double a_z = 0;
const long double b_z = 1;