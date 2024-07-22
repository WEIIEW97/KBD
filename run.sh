# Default mode is single directory
mode="single"
use_global="false"
root_dir="/Users/williamwei/Codes/KBD/data"

# Check for flags: '-m' for multiple directories and '-g' for global processing
while getopts "mg" opt; do
  case $opt in
    m) mode="multiple"
       ;;
    g) use_global="true"
       ;;
    *) echo "Usage: $0 [-m for multiple directories] [-g to use global flag]"
       exit 1
       ;;
  esac
done

process_directory() {
    local base_path="$1"
    local global_flag="$2"
    local file_path="$base_path/image_data"
    local csv_path=$(find $base_path -type f -name "depthquality*.csv" | head -n 1)
    local output_path="${base_path}/image_data_lc++"

    # Navigate to the build directory and execute the program
    cd build
    if [ "$global_flag" == "true" ]; then
        ./kbd -f "$file_path" -c "$csv_path" -t "$output_path" -g
    else
        ./kbd -f "$file_path" -c "$csv_path" -t "$output_path"
    fi
    cd - > /dev/null
}

if [ "$mode" == "multiple" ]; then
    # Process all directories
    for camera_type in $(ls -d $root_dir/*/); do
        echo "Processing directory $camera_type with global flag: $use_global"
        process_directory "$camera_type" "$use_global"
    done
else
    # Process only the specified directory
    specific_dir="/Users/williamwei/Codes/KBD/data/N09ASH24DH0050"
    echo "Processing single directory $specific_dir with global flag: $use_global"
    process_directory "$specific_dir" "$use_global"
fi
