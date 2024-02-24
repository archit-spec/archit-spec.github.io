#include <iostream>
using namespace std;

int main() {
    int new_element = 9;
    int num[] = {2, 8, 7, 6, 0};
    int n = sizeof(num) / sizeof(num[0]);
    n = n + 1;
    num[n - 1] = new_element;
    for (int i = 0; i < n; i++) {
        cout << num[i] << " ";
    }
    return 0;
}

