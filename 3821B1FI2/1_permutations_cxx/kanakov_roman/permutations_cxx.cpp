#include "permutations_cxx.h"

#include <unordered_map>
#include <algorithm>
#include <iterator>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> um;

  for (const auto& [str_, _] : dictionary) {
	std::string str = str_;
	std::sort(str.begin(), str.end());
	um[str].push_back(str);
  }

  for (auto& [sorted_str, original_strings] : um) {
	std::reverse(original_strings.begin(), original_strings.end());
	for (const auto& original_str : original_strings)
	  dictionary[original_str] = original_strings;
  }

  for (auto& [str, perms] : dictionary) {
	perms.erase(std::remove(perms.begin(), perms.end(), str),
	            perms.end());
  }
}