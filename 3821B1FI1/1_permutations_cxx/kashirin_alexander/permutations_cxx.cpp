#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> anagram_sets;

    std::for_each(dictionary.rbegin(), dictionary.rend(),
        [&anagram_sets](const auto& entry) {
            std::string normalized = entry.first;
            std::sort(normalized.begin(), normalized.end());
            anagram_sets[normalized].emplace_back(entry.first);
        }
    );

    for (const auto& [_, anagrams] : anagram_sets) {
        if (anagrams.size() <= 1) continue;

        for (const auto& word : anagrams) {
            auto& relations = dictionary[word];
            relations.reserve(relations.size() + anagrams.size() - 1);

            std::copy_if(anagrams.begin(), anagrams.end(),
                std::back_inserter(relations),
                [&word](const auto& anagram) { return anagram != word; }
            );
        }
    }
}
