#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include <fmt/core.h>
#include <memory>
#include <filesystem>
#include <iostream>

#include "src/table.h"
#include "src/config.h"
#include "src/array.h"
#include "src/format.h"

namespace fs = std::filesystem;

int main() {
  auto cwd = fs::current_path();
  // std::cout << cwd.parent_path().string() << std::endl;
  fmt::print("current working path is {}.\n", cwd.parent_path().string());
  std::string file_path =
      cwd.parent_path().string() + "/depthquality_2024-07-04.csv";

  auto default_config = kbd::Config();

  auto arrow_csv_reader = kbd::ArrowTableReader();
  auto original_table = arrow_csv_reader.read_csv(file_path);
  auto trimmed_table =
      arrow_csv_reader.trim_table(default_config.MAPPED_PAIR_DICT);

  auto table_names = trimmed_table->ColumnNames();
  for (const auto& name : table_names) {
    // std::cout << name << "\n";
    fmt::print("table name is {}\n", name);
  }

  auto avg_disp_col =
      trimmed_table->GetColumnByName(default_config.AVG_DISP_NAME);
  auto avg_disp_col_array =
      std::static_pointer_cast<arrow::DoubleArray>(avg_disp_col->chunk(0));
  auto eigen_disp_array = kbd::ArrowDoubleArrayToEigen<double>(avg_disp_col_array);
  fmt::print("working on conversion ... \n");
  fmt::print("converted eigen array: \n {} \n", eigen_disp_array);
  return 0;
}
