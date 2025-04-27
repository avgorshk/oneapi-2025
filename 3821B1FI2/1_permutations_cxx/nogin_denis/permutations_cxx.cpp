#include "permutations_cxx.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t &dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> permutations;

  for (const auto &[key, _] : dictionary) {
    std::string sortKey = key;
    std::sort(sortKey.begin(), sortKey.end());
    permutations[sortKey].push_back(key);
  }

  for (auto &[key, perm] : dictionary) {
    std::string sortKey = key;
    std::sort(sortKey.begin(), sortKey.end());

    if (permutations[sortKey].size() > 1) {
      perm = permutations[sortKey];
      perm.erase(std::remove(perm.begin(), perm.end(), key), perm.end());
      std::sort(perm.rbegin(), perm.rend());
    }
  }
}
