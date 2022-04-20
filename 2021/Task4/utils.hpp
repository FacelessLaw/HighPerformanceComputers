#include <cmath>
#include <fstream>
#include "config.hpp"

void domains_for_procs(
    int nprocs,
    vector<int> &istart,
    vector<int> &jstart,
    vector<int> &ilen,
    vector<int> &jlen,
    int *l_neighbours,
	int *r_neighbours,
	int *b_neighbours,
	int *t_neighbours
)
{
  	int power = 0;
	int nproc = nprocs;
	while (nproc > 1) {
		nproc /= 2;
		power += 1;
	}

	int x_num, y_num;
	y_num = (int)std::pow((double)2, power / 2);
	x_num = (int)std::pow((double)2, power / 2 + power % 2);

	std::vector<int> x_sizes(x_num, (M + 1) / x_num);
	for (int k = 0; k < (M + 1) % x_num; ++k)
		x_sizes[k] += 1;

	std::vector<int> y_sizes(y_num, (N + 1) / y_num);
	for (int k = 0; k < (N + 1) % y_num; ++k)
		y_sizes[k] += 1;

	int left_top_y = -1, right_bottom_y = -1;
	for (int i = 0; i < y_sizes.size(); ++i) {
		right_bottom_y += y_sizes[i];
		int left_top_x = -1, right_bottom_x = -1;
		for (int j = 0; j < x_sizes.size(); ++j) {
			right_bottom_x += x_sizes[j];
			int curr_proc = i * (int) x_sizes.size() + j;
			l_neighbours[curr_proc] = i * (int) x_sizes.size() + j - 1;
			r_neighbours[curr_proc] = i * (int) x_sizes.size() + j + 1;
			b_neighbours[curr_proc] = (i - 1) * (int) x_sizes.size() + j;
			t_neighbours[curr_proc] = (i + 1) * (int) x_sizes.size() + j;
			istart[curr_proc] = left_top_x + 1;
			jstart[curr_proc] = left_top_y + 1;
			ilen[curr_proc] = right_bottom_x - left_top_x;
			jlen[curr_proc] = right_bottom_y - left_top_y;
			left_top_x += x_sizes[j];
		}
		left_top_y += y_sizes[i];
	}
    return;
}
