#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> sorted_map;

    for (const auto& [str, _] : dictionary) {
        std::string sorted_str = str;
        std::sort(sorted_str.begin(), sorted_str.end());
        sorted_map[sorted_str].push_back(str);
    }

    for (auto& [sorted_str, original_strings] : sorted_map) {
        std::reverse(original_strings.begin(), original_strings.end());
        for (const auto& original_str : original_strings) {
            dictionary[original_str] = original_strings;
        }
    }

    for (auto& [str, perms] : dictionary) {
        perms.erase(
            std::remove(perms.begin(), perms.end(), str),
            perms.end());
    }
}