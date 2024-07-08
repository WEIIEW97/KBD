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

#ifndef KBD_CSV_H
#define KBD_CSV_H

#include "shared.h"

#include <memory>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
namespace kbd {
  class ArrowCSVReader {
  public:
    ArrowCSVReader() { initialize(); }
    ~ArrowCSVReader() = default;

    std::shared_ptr<arrow::Table> df_;

  private:
    arrow::io::IOContext io_context_;
    arrow::MemoryPool* mem_pool_;
    arrow::csv::ReadOptions read_options_;
    arrow::csv::ParseOptions parse_options_;
    arrow::csv::ConvertOptions convert_options_;

  public:
    void initialize();
    std::shared_ptr<arrow::Table> read_csv(const std::string& csv_path);
    std::shared_ptr<arrow::Schema> get_column_names(bool verbose=true);
    std::shared_ptr<arrow::ChunkedArray> get_data_by_column_name(const std::string& column_name);
    arrow::Status arrow_pretty_print(const arrow::ChunkedArray& data);
  };
} // namespace kbd

#endif // KBD_CSV_H
