#ifndef __CONFIG_HPP__ 
#define __CONFIG_HPP__

#include <vector>

using namespace std;

const int M = 500;
const int N = 500;

const double A1 = 0;
const double A2 = 4;
const double B1 = 0;
const double B2 = 3;

double h1 = (A2 - A1) / M;
double h2 = (B2 - B1) / N;

const double EPS = 1e-6;
const double TARGET_EPS = 5e-4;

#endif
