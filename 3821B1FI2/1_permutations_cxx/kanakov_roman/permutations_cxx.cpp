#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>
#include <iterator>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<dictionary_t::iterator>> tmp;
  for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
    std::string key = it->first;
    std::sort(key.begin(), key.end());
    tmp[key].push_back(it);
  }
  for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
    std::string key = it->first;
    std::sort(key.begin(), key.end());
    const std::vector<dictionary_t::iterator>& v = tmp.at(key);
    if (v.size() > 1) {
      std::vector<std::string>& v_res = it->second;
      for (auto v_it = v.rbegin(); v_it != v.rend(); v_it++) {
        const auto& _v_it = *v_it;
        if (_v_it != it) {
          v_res.push_back(_v_it->first);
        }
      }
    }
  }
}