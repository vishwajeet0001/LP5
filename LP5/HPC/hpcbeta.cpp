#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void linear_regression(const vector<double> &x, const vector<double> &y, double &b0, double &b1, double &t, bool parallel) {
    int n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    double start = omp_get_wtime();

    if (parallel) {
#pragma omp parallel for reduction(+ : sum_x, sum_y, sum_xy, sum_x2)
        for (int i = 0; i < n; i++) {
            sum_x += x[i]; sum_y += y[i];
            sum_xy += x[i] * y[i]; sum_x2 += x[i] * x[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            sum_x += x[i]; sum_y += y[i];
            sum_xy += x[i] * y[i]; sum_x2 += x[i] * x[i];
        }
    }

    b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    b0 = (sum_y - b1 * sum_x) / n;
    t = omp_get_wtime() - start;
}

int main() {
    int n = 1000000; // 10 million elements
    vector<double> x(n), y(n);

    for (int i = 0; i < n; ++i) {
        x[i] = i;
        y[i] = 3.0 * i + 2.0; // y = 3x + 2
    }

    omp_set_num_threads(omp_get_max_threads());

    double b0_s, b1_s, t_s, b0_p, b1_p, t_p;
    linear_regression(x, y, b0_s, b1_s, t_s, false);
    linear_regression(x, y, b0_p, b1_p, t_p, true);

    cout << "Sequential: y = " << b1_s << "x + " << b0_s << " | Time: " << t_s << "s\n";
    cout << "Parallel:   y = " << b1_p << "x + " << b0_p << " | Time: " << t_p << "s\n";
    cout << "Speedup: " << t_s / t_p << endl;
    return 0;
}
