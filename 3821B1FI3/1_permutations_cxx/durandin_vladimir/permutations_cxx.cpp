#include "permutations_cxx.h"

#include <algorithm>
#include <iostream>

void Permutations(dictionary_t &dictionary) {

  const auto &comparator = [](const std::string &_first,
                              const std::string &_second) -> bool {
    return _first > _second;
  };

  dictionary_t permutations;

  for (auto &[key, value] : dictionary) {
    std::string main_sort_key = key;
    std::sort(main_sort_key.begin(), main_sort_key.end());
    for (auto &[other_key, other_value] : dictionary) {
      std::string second_sort_key = other_key;
      std::sort(second_sort_key.begin(), second_sort_key.end());
      if (main_sort_key == second_sort_key && key != other_key) {
        permutations[key].push_back(other_key);
      } else {
        permutations[other_key];
      }
    }
    std::sort(permutations[key].begin(), permutations[key].end(), comparator);
  }
  dictionary = permutations;
}
