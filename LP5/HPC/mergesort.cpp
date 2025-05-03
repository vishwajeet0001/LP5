#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

void bubbleSortSequential(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
    cout << "The array elements after sequential bubble sort are:" << endl;

    for (int i = 0; i < 10; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void bubbleSortParallel(vector<int> &arr)
{
    int n = arr.size();
#pragma omp parallel for
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

void merge(vector<int> &arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; ++i)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

void mergeSortSequential(vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

void mergeSortParallel(vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            mergeSortParallel(arr, left, mid);

#pragma omp section
            mergeSortParallel(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

vector<int> generateRandomArray(int size)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 10000);

    vector<int> arr(size);
    for (int i = 0; i < size; ++i)
        arr[i] = dist(gen);
    return arr;
}

int main()
{
    int size;
    cout << "Enter the size of the array:";
    cin >> size;
    vector<int> arr = generateRandomArray(size);

    double time_seq_bubble, time_par_bubble, time_seq_merge, time_par_merge;

    vector<int> arr_bubble = arr;
    auto start = chrono::high_resolution_clock::now();
    bubbleSortSequential(arr_bubble);
    auto end = chrono::high_resolution_clock::now();
    time_seq_bubble = chrono::duration<double>(end - start).count();
    cout << "Time required for Bubble Sort sequential is: " << time_seq_bubble << " seconds" << endl;

    vector<int> arr_par_bubble = arr;
    start = chrono::high_resolution_clock::now();
    bubbleSortParallel(arr_par_bubble);
    end = chrono::high_resolution_clock::now();
    time_par_bubble = chrono::duration<double>(end - start).count();
    cout << "Time required for Bubble Sort parallel is: " << time_par_bubble << " seconds" << endl;

    vector<int> arr_merge = arr;
    start = chrono::high_resolution_clock::now();
    mergeSortSequential(arr_merge, 0, arr_merge.size() - 1);
    end = chrono::high_resolution_clock::now();
    time_seq_merge = chrono::duration<double>(end - start).count();
    cout << "Time required for Merge Sort sequential is: " << time_seq_merge << " seconds" << endl;

    vector<int> arr_par_merge = arr;
    start = chrono::high_resolution_clock::now();
    mergeSortParallel(arr_par_merge, 0, arr_par_merge.size() - 1);
    end = chrono::high_resolution_clock::now();
    time_par_merge = chrono::duration<double>(end - start).count();
    cout << "Time required for Merge Sort parallel is: " << time_par_merge << " seconds" << endl;

    cout << "Bubble Sort Speedup time: " << (time_par_bubble > 0 ? time_seq_bubble / time_par_bubble : 0) << endl;
    cout << "Merge Sort Speedup time: " << (time_par_merge > 0 ? time_seq_merge / time_par_merge : 0) << endl;

    return 0;
}
