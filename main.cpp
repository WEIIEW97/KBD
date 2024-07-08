#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include <iostream>
#include <memory>

int main() {
  arrow::io::IOContext io_context = arrow::io::default_io_context();
  // Memory pool used by Arrow to efficiently allocate and deallocate memory
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::string file_path =
      "/Users/williamwei/Codes/KBD/depthquality_2024-07-04.csv";
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
  auto status = arrow::PrettyPrint(*column, {}, &std::cout);

  return 0;
}
