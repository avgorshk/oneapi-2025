#include "permutations_cxx.h"

#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> grouped;

    for (const auto& entry : dictionary) {
        std::string sorted_key = entry.first;
        std::sort(sorted_key.begin(), sorted_key.end());
        grouped[sorted_key].push_back(entry.first);
    }

    for (auto& entry : dictionary) {
        std::string sorted_key = entry.first;
        std::sort(sorted_key.begin(), sorted_key.end());
        std::vector<std::string>& permutations = grouped[sorted_key];

        permutations.erase(std::remove(permutations.begin(), permutations.end(), entry.first), permutations.end());

        std::sort(permutations.begin(), permutations.end(), std::greater<std::string>());

        entry.second = permutations;
    }
}
