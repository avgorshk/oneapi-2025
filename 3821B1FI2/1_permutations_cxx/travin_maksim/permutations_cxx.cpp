#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>


void Permutations(dictionary_t& dictionary) {
    dictionary_t temp_map;
    temp_map.reserve(dictionary.size());
    for (const auto& [s, v] : dictionary) {
        std::string sorted_key = s;
        std::sort(sorted_key.begin(), sorted_key.end());
        temp_map[sorted_key].push_back(s);
    }

    for (auto& [s, v] : dictionary) {
        std::string sorted_key = s;
        std::sort(sorted_key.begin(), sorted_key.end());

        auto it = temp_map.find(sorted_key);
        v.reserve(it->second.size());
        for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit) {
            if (*rit != s) {
                v.push_back(*rit);
            }
        }
    }
}
