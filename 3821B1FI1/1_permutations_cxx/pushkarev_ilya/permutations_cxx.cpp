#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> groups;

    for (const auto& [word, _] : dictionary) {
        std::string key = word;
        std::sort(key.begin(), key.end());
        groups[key].push_back(word);
    }

    for (const auto& pair : groups) {
        const auto& anagrams = pair.second;
        if (anagrams.size() < 2) continue;

        for (const auto& word : anagrams) {
            auto& related = dictionary[word];
            related.reserve(related.size() + anagrams.size() - 1);

            for (const auto& candidate : anagrams) {
                if (candidate != word) {
                    related.push_back(candidate);
                }
            }
        }
    }
}
