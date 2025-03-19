#include "permutations_cxx.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t &dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> permutations;

  for (const auto &[key, _] : dictionary) {
    std::string sortedKey = key;
    std::sort(sortedKey.begin(), sortedKey.end());
    permutations[sortedKey].push_back(key);
  }

  for (auto &[key, perm] : dictionary) {
    std::string sortedKey = key;
    std::sort(sortedKey.begin(), sortedKey.end());

    if (permutations[sortedKey].size() > 1) {
      perm = permutations[sortedKey];
      perm.erase(std::remove(perm.begin(), perm.end(), sortedKey), perm.end());
      std::sort(perm.rbegin(), perm.rend());
    }
  }
}
