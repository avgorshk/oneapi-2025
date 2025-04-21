#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& d) {
    std::unordered_map<std::string, std::vector<std::string>> g;

    for (const auto& p : d) {
        auto s = p.first;
        std::sort(s.begin(), s.end());
        g[s].push_back(p.first);
    }

    for (const auto& p : g) {
        for (const auto& x : p.second) {
            auto& v = d[x];
            v.clear();
            for (const auto& y : p.second)
                if (y != x) v.push_back(y);
            std::sort(v.rbegin(), v.rend());
        }
    }
}
