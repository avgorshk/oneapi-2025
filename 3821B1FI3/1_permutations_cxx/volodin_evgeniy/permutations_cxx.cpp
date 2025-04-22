#include "permutations_cxx.h"
#include <algorithm>

void Permutations(dictionary_t &dictionary) {
  auto sortString = [](const std::string &str) {
    std::string sorted = str;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
  };

  dictionary_t grouped;

  for (const auto &pair : dictionary) {
    grouped[sortString(pair.first)].push_back(pair.first);
  }

  for (auto &pair : dictionary) {
    const std::string &key = pair.first;
    const auto &permutations = grouped[sortString(key)];

    pair.second.reserve(permutations.size() - 1);
    for (const auto &perm : permutations) {
      if (perm != key) {
        pair.second.push_back(perm);
      }
    }

    std::sort(pair.second.rbegin(), pair.second.rend());
  }
}
