#include "permutations_cxx.h"

#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    dictionary_t perm;
    for (auto& [s, v] : dictionary) {
        std::string s_sorted = s;
        std::sort(s_sorted.begin(), s_sorted.end());
        perm[s_sorted].push_back(s);
    }
    for (auto& [s, v] : dictionary) {
        std::string s_sorted = s;
        std::sort(s_sorted.begin(), s_sorted.end());
        auto& V = perm[s_sorted];
        for (auto it = V.rbegin(); it != V.rend(); it++) {
            if (*it != s) {
                v.push_back(*it);
            }
        }
    }
}
