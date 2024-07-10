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

#ifndef KBD_TABLE_H
#define KBD_TABLE_H

#include "config.h"

#include <memory>
#include <map>
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include <arrow/util/logging.h>

namespace kbd {
  class ArrowTableReader {
  public:
    ArrowTableReader() { initialize(); }
    ~ArrowTableReader() = default;

    std::shared_ptr<arrow::Table> df_;
    double baseline_ = 0.f;
    double focal_ = 0.f;

  private:
    arrow::io::IOContext io_context_;
    arrow::MemoryPool* mem_pool_;
    arrow::csv::ReadOptions read_options_;
    arrow::csv::ParseOptions parse_options_;
    arrow::csv::ConvertOptions convert_options_;

  public:
    void initialize();
    std::shared_ptr<arrow::Table> read_csv(const std::string& csv_path);
    std::shared_ptr<arrow::Schema> get_column_names(bool verbose = true);
    std::shared_ptr<arrow::ChunkedArray>
    get_data_by_column_name(const std::string& column_name);
    arrow::Status arrow_pretty_print(const arrow::ChunkedArray& data);
    std::shared_ptr<arrow::Table>
    trim_table(const std::map<std::string, std::string>& pair_dict);
    arrow::Status map_table(std::shared_ptr<arrow::Table> table,
                            const Config& kbd_config,
                            const std::map<std::string, double>& dist_dict);
  };
} // namespace kbd

#endif // KBD_TABLE_H
