#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    // Создаем временное хранилище для группировки слов по анаграммам
    std::unordered_map<std::string, std::vector<std::string>> anagram_sets;
    
    // Формируем группы анаграмм
    std::for_each(dictionary.rbegin(), dictionary.rend(), 
        [&anagram_sets](const auto& entry) {
            std::string normalized = entry.first;
            std::sort(normalized.begin(), normalized.end());
            anagram_sets[normalized].emplace_back(entry.first);
        }
    );
    
    // Заполняем словарь связями между анаграммами
    for (const auto& [_, anagrams] : anagram_sets) {
        if (anagrams.size() <= 1) continue;
        
        // Для каждого слова добавляем все его анаграммы, кроме самого себя
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