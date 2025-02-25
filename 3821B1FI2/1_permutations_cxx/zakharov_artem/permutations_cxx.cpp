#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

using hash_dictionary_t = std::unordered_map<std::string, std::vector<std::string>>;


void Permutations(dictionary_t& dictionary) {
    hash_dictionary_t tmp_dict;
    for (auto it = dictionary.crbegin(); it != dictionary.crend(); ++it) {
        auto& [perm, _] = *it;
        std::string key = perm;
        std::sort(key.begin(), key.end());
        tmp_dict[key].push_back(perm);
    }

    for (auto& [_, perms] : tmp_dict) {
        size_t n = perms.size();
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                if (i == j) {
                    continue;
                }
                dictionary[perms[i]].push_back(perms[j]);
            }
        }
    }
}
