#include <iostream>
#include <omp.h>
#include <vector>
#include <climits>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

void generateArray(vector<int> &arr, int n)
{
    for (int &num : arr)
    {
        num = rand() % 100;
    }
}

double minSequential(const vector<int> &arr, int &min_val)
{
    auto start = high_resolution_clock::now();
    min_val = INT_MAX;
    for (int num : arr)
    {
        if (num < min_val)
            min_val = num;
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double minParallel(const vector<int> &arr, int &min_val)
{
    auto start = high_resolution_clock::now();
    min_val = INT_MAX;
#pragma omp parallel for reduction(min : min_val)
    for (size_t i = 0; i < arr.size(); i++)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double maxSequential(const vector<int> &arr, int &max_val)
{
    auto start = high_resolution_clock::now();
    max_val = INT_MIN;
    for (int num : arr)
    {
        if (num > max_val)
            max_val = num;
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double maxParallel(const vector<int> &arr, int &max_val)
{
    auto start = high_resolution_clock::now();
    max_val = INT_MIN;
#pragma omp parallel for reduction(max : max_val)
    for (size_t i = 0; i < arr.size(); i++)
    {
        if (arr[i] > max_val)
            max_val = arr[i];
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double sumSequential(const vector<int> &arr, long long &sum_val)
{
    auto start = high_resolution_clock::now();
    sum_val = 0;
    for (int num : arr)
    {
        sum_val += num;
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double sumParallel(const vector<int> &arr, long long &sum_val)
{
    auto start = high_resolution_clock::now();
    sum_val = 0;
#pragma omp parallel for reduction(+ : sum_val)
    for (size_t i = 0; i < arr.size(); i++)
    {
        sum_val += arr[i];
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double avgSequential(const vector<int> &arr, double &avg_val)
{
    long long sum_val;
    double time = sumSequential(arr, sum_val);
    avg_val = static_cast<double>(sum_val) / arr.size();
    return time;
}

double avgParallel(const vector<int> &arr, double &avg_val)
{
    long long sum_val;
    double time = sumParallel(arr, sum_val);
    avg_val = static_cast<double>(sum_val) / arr.size();
    return time;
}

int main()
{
    omp_set_num_threads(4);

    int n;
    cout << "Enter the number of elements in the array: ";
    cin >> n;
    vector<int> arr(n);

    generateArray(arr, n);

    cout << "\nArray elements are: ";
    for (int i = 0; i < min(n, 10); ++i)
        cout << arr[i] << " ";
    cout << endl;

    int min_val, max_val;
    long long sum_val;
    double avg_val;

    double t_min_seq = minSequential(arr, min_val);
    cout << "\nSequential Min Value: " << min_val << "\nTime for Sequential Min: " << t_min_seq << " seconds";

    double t_min_par = minParallel(arr, min_val);
    cout << "\nParallel Min Value: " << min_val << "\nTime for Parallel Min: " << t_min_par << " seconds";
    cout << "\nSpeedup for Min: " << (t_min_seq / t_min_par) << endl;

    double t_max_seq = maxSequential(arr, max_val);
    cout << "\nSequential Max Value: " << max_val << "\nTime for Sequential Max: " << t_max_seq << " seconds";

    double t_max_par = maxParallel(arr, max_val);
    cout << "\nParallel Max Value: " << max_val << "\nTime for Parallel Max: " << t_max_par << " seconds";
    cout << "\nSpeedup for Max: " << (t_max_seq / t_max_par) << endl;

    double t_sum_seq = sumSequential(arr, sum_val);
    cout << "\nSequential Sum: " << sum_val << "\nTime for Sequential Sum: " << t_sum_seq << " seconds";

    double t_sum_par = sumParallel(arr, sum_val);
    cout << "\nParallel Sum: " << sum_val << "\nTime for Parallel Sum: " << t_sum_par << " seconds";
    cout << "\nSpeedup for Sum: " << (t_sum_seq / t_sum_par) << endl;

    double t_avg_seq = avgSequential(arr, avg_val);
    cout << "\nSequential Average: " << avg_val << "\nTime for Sequential Average: " << t_avg_seq << " seconds";

    double t_avg_par = avgParallel(arr, avg_val);
    cout << "\nParallel Average: " << avg_val << "\nTime for Parallel Average: " << t_avg_par << " seconds";
    cout << "\nSpeedup for Average: " << (t_avg_seq / t_avg_par) << endl;

    return 0;
}
