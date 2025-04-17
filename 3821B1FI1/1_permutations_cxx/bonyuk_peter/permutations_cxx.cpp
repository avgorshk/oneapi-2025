#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> anagram_sets;
    
    for (dictionary_t::reverse_iterator it = dictionary.rbegin(); it != dictionary.rend(); ++it) {
        std::string normalized = it->first;
        std::sort(normalized.begin(), normalized.end());
        anagram_sets[normalized].push_back(it->first);
    }
    
    for (auto it = anagram_sets.begin(); it != anagram_sets.end(); ++it) {
        const std::vector<std::string>& anagrams = it->second;
        if (anagrams.size() <= 1) continue;
        
        for (size_t i = 0; i < anagrams.size(); ++i) {
            const std::string& word = anagrams[i];
            std::vector<std::string>& relations = dictionary[word];
            relations.reserve(relations.size() + anagrams.size() - 1);
            
            for (size_t j = 0; j < anagrams.size(); ++j) {
                if (i != j) {
                    relations.push_back(anagrams[j]);
                }
            }
        }
    }
}