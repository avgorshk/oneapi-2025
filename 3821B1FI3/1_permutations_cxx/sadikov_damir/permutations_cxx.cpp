#include "permutations_cxx.h"

#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    for (auto& [s1, v1] : dictionary) {
        for (auto& [s2, v2] : dictionary) {
            if (s1 != s2 && std::is_permutation(s1.begin(), s1.end(), s2.begin(), s2.end())) {
                v1.push_back(s2);
            }
        }
    }
    for (auto& [s, v] : dictionary) {
        std::reverse(v.begin(), v.end());
    }
}
