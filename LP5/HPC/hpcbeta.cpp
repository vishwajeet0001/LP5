#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void sequential_lr(const vector<double> &x, const vector<double> &y, double &beta0, double &beta1, double &time)
{
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

    double start = omp_get_wtime();

    for (int i = 0; i < n; ++i)
    {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    beta0 = (sum_y - beta1 * sum_x) / n;

    double end = omp_get_wtime();
    time = end - start;
}

void parallel_lr(const vector<double> &x, const vector<double> &y, double &beta0, double &beta1, double &time)
{
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

    double start = omp_get_wtime();

#pragma omp parallel for reduction(+ : sum_x, sum_y, sum_xy, sum_x2)
    for (int i = 0; i < n; ++i)
    {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    beta0 = (sum_y - beta1 * sum_x) / n;

    double end = omp_get_wtime();
    time = end - start;
}

int main()
{
    int n = 100000; // 10 million elements
    vector<double> x(n), y(n);

    // for (int i = 0; i < n; ++i)
    // {
    //     x[i] = i;
    //     y[i] = 3.0 * i + 2.0; // y = 3x + 2
    // }
    x = {
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    };
    y = {2.0, 4.0, 5.0, 4.0, 5.0};

    omp_set_num_threads(omp_get_max_threads()); // use max threads

    double beta0_seq, beta1_seq, time_seq;
    double beta0_par, beta1_par, time_par;

    sequential_lr(x, y, beta0_seq, beta1_seq, time_seq);
    parallel_lr(x, y, beta0_par, beta1_par, time_par);

    cout << "\nSequential Execution:\n";
    cout << "beta0: " << beta0_seq << ", beta1: " << beta1_seq << ", Time: " << time_seq << "s\n";
    cout << "Line Equation (Sequential): y = " << beta1_seq << "x + " << beta0_seq << endl;

    cout << "\nParallel Execution:\n";
    cout << "beta0: " << beta0_par << ", beta1: " << beta1_par << ", Time: " << time_par << "s\n";
    cout << "Line Equation (Parallel): y = " << beta1_par << "x + " << beta0_par << endl;

    double speedup = time_seq / time_par;
    cout << "\nSpeedup: " << speedup << endl;

    return 0;
}
