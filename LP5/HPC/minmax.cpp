#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Parallel Min Reduction
int parallel_min(const vector<int>& arr) {
    int min_value = arr[0];
    #pragma omp parallel for reduction(min:min_value)
    for (int i = 1; i < arr.size(); i++) {
        min_value = min(min_value, arr[i]);
    }
    return min_value;
}

// Parallel Max Reduction
int parallel_max(const vector<int>& arr) {
    int max_value = arr[0];
    #pragma omp parallel for reduction(max:max_value)
    for (int i = 1; i < arr.size(); i++) {
        max_value = max(max_value, arr[i]);
    }
    return max_value;
}

// Parallel Sum Reduction
int parallel_sum(const vector<int>& arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

// Parallel Average Calculation (using sum and size)
double parallel_average(const vector<int>& arr) {
    int sum = parallel_sum(arr);
    return static_cast<double>(sum) / arr.size();
}

int main() {
    // Example vector
    vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Parallel Min, Max, Sum, and Average
    int min_value = parallel_min(arr);
    int max_value = parallel_max(arr);
    int sum_value = parallel_sum(arr);
    double avg_value = parallel_average(arr);

    // Output results
    cout << "Parallel Min: " << min_value << endl;
    cout << "Parallel Max: " << max_value << endl;
    cout << "Parallel Sum: " << sum_value << endl;
    cout << "Parallel Average: " << avg_value << endl;

    return 0;
}
