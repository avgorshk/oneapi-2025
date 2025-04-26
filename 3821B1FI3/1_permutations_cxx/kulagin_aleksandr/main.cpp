// Copyright (c) 2025 Kulagin Aleksandr
#include "permutations_cxx.h"
#include <sstream>
#include <iostream>

static std::vector<std::string> split(const std::string& str, char del = ' ') {
  std::vector<std::string> res;
  std::stringstream ss(str);
  std::string tmp;
  while (std::getline(ss, tmp, del)) {
    res.push_back(tmp);
  }
  return res;
}

int main(int argc, char* argv[]) {
  std::string test_str = R"(aaa
acb
acd
ad
adc
bac
bc
bcc
bd
bda
bdc
caa
cad
cb
cc
ccb
cd
dac
db
dc
dca
dcb
dcc
dd)";
  std::vector<std::string> test_str_splitted = split(test_str, '\n');
  dictionary_t test_map;
  for (const auto& _test_str : test_str_splitted) {
    test_map[_test_str] = std::vector<std::string>();
  }
  Permutations(test_map);
  for (const auto& it : test_map) {
    std::cout << it.first << " :";
    for (const auto& v_it : it.second) {
      std::cout << ' ' << v_it;
    }
    std::cout << '\n';
  }
  return 0;
}
