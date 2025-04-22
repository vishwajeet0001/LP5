#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>
#include <algorithm>

using namespace std;

// Sequential Bubble Sort
void bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) swap(arr[j], arr[j+1]);
        }
    }
}

// Parallel Bubble Sort
void parallel_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) swap(arr[j], arr[j+1]);
        }
    }
}

// Merge function for Merge Sort
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> left_arr(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> right_arr(arr.begin() + mid + 1, arr.begin() + right + 1);
    int i = 0, j = 0, k = left;
    while (i < left_arr.size() && j < right_arr.size()) {
        arr[k++] = (left_arr[i] <= right_arr[j]) ? left_arr[i++] : right_arr[j++];
    }
    while (i < left_arr.size()) arr[k++] = left_arr[i++];
    while (j < right_arr.size()) arr[k++] = right_arr[j++];
}

// Sequential Merge Sort
void merge_sort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort
void parallel_merge_sort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(arr, left, mid);
            #pragma omp section
            parallel_merge_sort(arr, mid + 1, right);
        }
        merge(arr, left, mid, right);
    }
}

// Time measuring utility
double measure_time(void (*sort_func)(vector<int>&), vector<int>& arr) {
    clock_t start = clock();
    sort_func(arr);
    return double(clock() - start) / CLOCKS_PER_SEC;
}

int main() {
    int size = 10000;
    vector<int> arr(size), arr_copy;

    // Generate random data
    for (int& x : arr) x = rand() % 10000;

    // Sequential Bubble Sort
    arr_copy = arr;
    cout << "Sequential Bubble Sort Time: " << measure_time(bubble_sort, arr_copy) << " sec\n";

    // Parallel Bubble Sort
    arr_copy = arr;
    cout << "Parallel Bubble Sort Time: " << measure_time(parallel_bubble_sort, arr_copy) << " sec\n";

    // Sequential Merge Sort
    arr_copy = arr;
    cout << "Sequential Merge Sort Time: " << measure_time([](vector<int>& arr){ merge_sort(arr, 0, arr.size() - 1); }, arr_copy) << " sec\n";

    // Parallel Merge Sort
    arr_copy = arr;
    cout << "Parallel Merge Sort Time: " << measure_time([](vector<int>& arr){ parallel_merge_sort(arr, 0, arr.size() - 1); }, arr_copy) << " sec\n";

    return 0;
}
