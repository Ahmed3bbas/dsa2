 #include <iostream>
#include <memory>
#include <cassert>
#include <stdint.h>
#include <chrono>
#include <sstream>
#include <limits>
#include <cassert>
#include <numeric>
#include <cstdlib>

#include <arrow/ipc/feather.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/builder.h>
#include <bitset>
#include <arrow/io/interfaces.h>
#include <unordered_map>
#include "feature_calculator.h"
using namespace std;

class PrevClickTimeDelta : public GroupedFeatureCalculator<uint16_t, arrow::UInt16Type> {
public:
  PrevClickTimeDelta(): GroupedFeatureCalculator() {}
  
  virtual string name() {
    return "PrevClickTimeDelta";
  }
  
  virtual vector<uint16_t> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    vector<uint16_t> feature(ip.size());
    for (const auto &entry : grouped_click_times) {
      const auto &group_click_times = entry.second;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        if (index > 0) {
          const size_t curr_index = group_click_times[index];
          const size_t prev_index = group_click_times[index - 1];
          // if (index + 1 < group_click_times.size() && click_time[group_click_times[index + 1]] == click_time[curr_index]) {
          //   feature[curr_index] = 0;
          // } else {
            assert(click_time[curr_index] >= click_time[prev_index]);
            const uint64_t delta = (click_time[curr_index] - click_time[prev_index]) / (1000 * 1000 * 1000);
            feature[group_click_times[index]] = min<uint64_t>(delta, numeric_limits<uint16_t>::max() - 1);
          // }
        } else {
          feature[group_click_times[index]] = numeric_limits<uint16_t>::max();
        }
      }
    }
    return feature;
  }
};

int main(int argc, char **argv)
{

  if (argc != 5) {
    fprintf(stderr, "Usage: %s (input_train_feather) (input_test_feather)"
            " (output_train_feather) (output_test_feather)\n", argv[0]);
    exit(-1);
  }
  
  PrevClickTimeDelta feature_calculator;
  arrow::Status status =feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4]);
  assert(status.ok());
}
