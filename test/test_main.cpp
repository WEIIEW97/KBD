/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
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

#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include <iostream>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
  arrow::io::IOContext io_context = arrow::io::default_io_context();
  // Memory pool used by Arrow to efficiently allocate and deallocate memory
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  auto cwd = fs::current_path();
  std::cout << cwd.parent_path().string() << std::endl;
  std::string file_path = cwd.parent_path().string() + "/depthquality_2024-07-04.csv";
  arrow::Result<std::shared_ptr<arrow::io::InputStream>> file_result =
      arrow::io::ReadableFile::Open(file_path, pool);
  if (!file_result.ok()) {
    std::cerr << "Failed to open file: " << file_result.status() << std::endl;
    return -1;
  }
  std::shared_ptr<arrow::io::InputStream> input = *file_result;

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  // Instantiate TableReader from input stream and options
  auto maybe_reader = arrow::csv::TableReader::Make(
      io_context, input, read_options, parse_options, convert_options);
  if (!maybe_reader.ok()) {
    // Handle TableReader instantiation error...
  }
  std::shared_ptr<arrow::csv::TableReader> reader = *maybe_reader;

  // Read table from CSV file
  auto maybe_table = reader->Read();
  if (!maybe_table.ok()) {
    // Handle CSV read error
    // (for example a CSV syntax error or failed type conversion)
  }
  std::shared_ptr<arrow::Table> table = *maybe_table;
  // Output the column names
  std::shared_ptr<arrow::Schema> schema = table->schema();
  std::cout << "Column Names:" << std::endl;
  for (const auto& field : schema->fields()) {
    std::cout << field->name() << std::endl;
  }

  std::string column_name = "fit plane dist/mm";
  std::shared_ptr<arrow::ChunkedArray> column =
      table->GetColumnByName(column_name);
  if (column == nullptr) {
    std::cerr << "Column not found: " << column_name << std::endl;
    return -1;
  }

  // Optionally print the column data
  std::cout << "Data in column '" << column_name << "':" << std::endl;
  auto status = arrow::PrettyPrint(*column->chunk(0), {}, &std::cout);
  return 0;
}
