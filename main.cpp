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
#include "src/utils.h"

namespace fs = std::filesystem;

int main() {
  std::string root_path = "/home/william/Codes/KBD/data/N09ASH24DH0050";
  std::string file_path = root_path + "/image_data";
  std::string csv_path = "/home/william/Codes/KBD/data/N09ASH24DH0050/depthquality_2024-07-09.csv";
  auto default_config = kbd::Config();

  auto arrow_csv_reader = kbd::ArrowTableReader();
  auto original_table = arrow_csv_reader.read_csv(csv_path);
  auto trimmed_table =
      arrow_csv_reader.trim_table(default_config.MAPPED_PAIR_DICT);
  auto table_names = trimmed_table->ColumnNames();
  for (const auto& name : table_names) {
    // std::cout << name << "\n";
    fmt::print("table name is {}\n", name);
  }

  auto column = trimmed_table->GetColumnByName(default_config.GT_DIST_NAME);
  auto column_int = std::static_pointer_cast<arrow::Int64Array>(column->chunk(0));
//  auto status1 = arrow::PrettyPrint(*column_int, {}, &std::cout);
  for (auto i = 0; i < column_int->length(); ++i) {
    auto v = column_int->Value(i);
    std::cout << v << "\n";
  }
  auto folders = kbd::retrieve_folder_names(file_path);
  auto dist_dict = kbd::calculate_mean_value(file_path, folders, default_config);

  auto status = arrow_csv_reader.map_table(trimmed_table, default_config, dist_dict);

  auto table_names1 = trimmed_table->ColumnNames();
  for (const auto& name : table_names1) {
    // std::cout << name << "\n";
    fmt::print("after mapping table name is {}\n", name);
  }


  auto avg_disp_col =
      trimmed_table->GetColumnByName(default_config.AVG_DISP_NAME);
  auto avg_disp_col_array =
      std::static_pointer_cast<arrow::DoubleArray>(avg_disp_col->chunk(0));
  auto eigen_disp_array_map = kbd::ArrowDoubleArrayToEigenMap<double>(avg_disp_col_array);
  Eigen::ArrayXd eigen_disp_array = eigen_disp_array_map;
  fmt::print("working on conversion ... \n");
  fmt::print("converted eigen array: \n {} \n", eigen_disp_array);
  return 0;
}
