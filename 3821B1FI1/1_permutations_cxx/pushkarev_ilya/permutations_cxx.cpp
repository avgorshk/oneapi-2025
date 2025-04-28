#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<dictionary_t::iterator>> grouped;
  
  for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
    auto key = it->first;
    std::sort(key.begin(), key.end());
    grouped[key].push_back(it);
  }

  for (auto& [word, anagrams] : dictionary) {
    auto key = word;
    std::sort(key.begin(), key.end());
    const auto& group = grouped.at(key);

    if (group.size() > 1) {
      auto& result = anagrams;
      for (auto rit = group.rbegin(); rit != group.rend(); ++rit) {
        if (*rit != dictionary.find(word)) {
          result.push_back((*rit)->first);
        }
      }
    }
  }
}
