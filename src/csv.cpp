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
#include "csv.h"

namespace kbd {
  void ArrowCSVReader::initialize() {
    io_context_ = arrow::io::default_io_context();
    mem_pool_ = arrow::default_memory_pool();
    read_options_ = arrow::csv::ReadOptions::Defaults();
    parse_options_ = arrow::csv::ParseOptions::Defaults();
    convert_options_ = arrow::csv::ConvertOptions::Defaults();
  }

  std::shared_ptr<arrow::Table>
  ArrowCSVReader::read_csv(const std::string& csv_path) {
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
  ArrowCSVReader::get_column_names(bool verbose) {
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
  ArrowCSVReader::get_data_by_column_name(const std::string& column_name) {
    std::shared_ptr<arrow::ChunkedArray> column =
        df_->GetColumnByName(column_name);
    if (column == nullptr) {
      std::cerr << "Column not found: " << column << std::endl;
    }
    return column;
  }

  arrow::Status
  ArrowCSVReader::arrow_pretty_print(const arrow::ChunkedArray& data) {
    auto status = arrow::PrettyPrint(data, {}, &std::cout);
    return status;
  }
} // namespace kbd