/*
 * Copyright (c) 2023--present, WILLIAM WEI.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "table.h"

#include <arrow/array.h>
#include <arrow/type.h>
namespace kbd {
  void ArrowTableReader::initialize() {
    io_context_ = arrow::io::default_io_context();
    mem_pool_ = arrow::default_memory_pool();
    read_options_ = arrow::csv::ReadOptions::Defaults();
    parse_options_ = arrow::csv::ParseOptions::Defaults();
    convert_options_ = arrow::csv::ConvertOptions::Defaults();
  }

  std::shared_ptr<arrow::Table>
  ArrowTableReader::read_csv(const std::string& csv_path) {
    arrow::Result<std::shared_ptr<arrow::io::InputStream>> file_result =
        arrow::io::ReadableFile::Open(csv_path, mem_pool_);
    if (!file_result.ok()) {
      std::cerr << "Failed to open file: " << file_result.status() << std::endl;
    }
    auto input = *file_result;
    auto maybe_reader = arrow::csv::TableReader::Make(
        io_context_, input, read_options_, parse_options_, convert_options_);
    if (!maybe_reader.ok()) {
      std::cerr << "Failed to create reader: " << maybe_reader.status()
                << std::endl;
    }
    auto reader = *maybe_reader;
    df_ = *reader->Read();

    return df_;
  }

  std::shared_ptr<arrow::Schema>
  ArrowTableReader::get_column_names(bool verbose) {
    std::shared_ptr<arrow::Schema> schema = df_->schema();
    if (verbose) {
      std::cout << "Column Names: " << std::endl;
      for (const auto& field : schema->fields()) {
        std::cout << field->name() << "\n";
      }
    }
    return schema;
  }

  std::shared_ptr<arrow::ChunkedArray>
  ArrowTableReader::get_data_by_column_name(const std::string& column_name) {
    std::shared_ptr<arrow::ChunkedArray> column =
        df_->GetColumnByName(column_name);
    if (column == nullptr) {
      std::cerr << "Column not found: " << column << std::endl;
    }
    return column;
  }

  arrow::Status
  ArrowTableReader::arrow_pretty_print(const arrow::ChunkedArray& data) {
    auto status = arrow::PrettyPrint(data, {}, &std::cout);
    return status;
  }

  std::shared_ptr<arrow::Table> ArrowTableReader::trim_table(
      const std::map<std::string, std::string>& pair_dict) {
    // selecting specific columns and renaming them
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;

    for (const auto& pair : pair_dict) {
      auto col = df_->GetColumnByName(pair.first);
      if (col != nullptr) {
        fields.push_back(arrow::field(pair.second, col->type()));
        columns.push_back(col->chunk(0));
      }
    }

    // create new schema with renamed columns
    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table_trimmed = arrow::Table::Make(schema, columns);

    return table_trimmed;
  }

  arrow::Status
  ArrowTableReader::map_table(std::shared_ptr<arrow::Table>& table,
                              const Config& kbd_config,
                              const std::map<std::string, double>& dist_dict) {
    auto gt_dist_col = table->GetColumnByName(kbd_config.GT_DIST_NAME);
    auto focal_col = table->GetColumnByName(kbd_config.FOCAL_NAME);
    auto baseline_col = table->GetColumnByName(kbd_config.BASELINE_NAME);

    if (!gt_dist_col || !focal_col || !baseline_col) {
      throw std::runtime_error("Required columns are missing in the table");
    }

    auto gt_dist_array = std::static_pointer_cast<arrow::Int64Array>(
        gt_dist_col->chunk(0)); // it must be int64 type
    auto focal_array =
        std::static_pointer_cast<arrow::DoubleArray>(focal_col->chunk(0));
    auto baseline_array =
        std::static_pointer_cast<arrow::DoubleArray>(baseline_col->chunk(0));

    // Extract the focal and baseline values (assuming they are the same for all
    // rows)
    focal_ = focal_array->Value(0);
    baseline_ = baseline_array->Value(0);

    arrow::DoubleBuilder avg_dist_builder, avg_disp_builder;
    auto fb = focal_ * baseline_;
    for (auto i = 0; i < gt_dist_array->length(); ++i) {
      auto gt_dist_value = gt_dist_array->Value(i);
      double avg_dist_value = dist_dict.at(std::to_string(gt_dist_value));
      ARROW_RETURN_NOT_OK(avg_dist_builder.Append(avg_dist_value));

      double avg_disp_value = fb / avg_dist_value;
      ARROW_RETURN_NOT_OK(avg_disp_builder.Append(avg_disp_value));
    }

    std::shared_ptr<arrow::Array> avg_dist_array, avg_disp_array;
    ARROW_RETURN_NOT_OK(avg_dist_builder.Finish(&avg_dist_array));
    ARROW_RETURN_NOT_OK(avg_disp_builder.Finish(&avg_disp_array));

    // Add new columns to the table
    auto schema_fields = table->schema()->fields();
    auto columns = table->columns();

    schema_fields.push_back(
        arrow::field(kbd_config.AVG_DIST_NAME, arrow::float64()));
    schema_fields.push_back(
        arrow::field(kbd_config.AVG_DISP_NAME, arrow::float64()));

    columns.push_back(std::make_shared<arrow::ChunkedArray>(avg_dist_array));
    columns.push_back(std::make_shared<arrow::ChunkedArray>(avg_disp_array));

    table = arrow::Table::Make(
        std::make_shared<arrow::Schema>(arrow::Schema(schema_fields)), columns);
    return arrow::Status::OK();
  }

} // namespace kbd