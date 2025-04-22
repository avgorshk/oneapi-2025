#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> grouped;

    for (const auto& [key, _] : dictionary) {
        std::string sorted_key = key;
        std::sort(sorted_key.begin(), sorted_key.end());
        grouped[sorted_key].push_back(key);
    }

    for (auto& [key, values] : dictionary) {
        std::string sorted_key = key;
        std::sort(sorted_key.begin(), sorted_key.end());

        auto& anagrams = grouped[sorted_key];
        values = anagrams;
        values.erase(std::remove(values.begin(), values.end(), key), values.end());
        std::sort(values.rbegin(), values.rend());
    }
}