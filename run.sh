# Default mode is single directory
mode="single"
use_global="false"
root_dir="/home/william/extdisk/data/KBD_ACCURACY"
log_file="$root_dir/logfile.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $@" | tee -a "$log_file"
}

# uncomment below lines if you are using bash
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

# this for zsh
# zparseopts -E -m:=mode -g:=use_global

process_directory() {
    local base_path="$1"
    local global_flag="$2"
    local file_path="$(realpath "${base_path}/image_data")"
    local csv_path=$(find $base_path -type f -name "depthquality*.csv" | head -n 1)
    local output_path="$(realpath "${base_path}/image_data_lc++")"

    if [[ -n "$csv_path" ]]; then
        echo "find csv!"
    else
        local xlsx_path=$(find $base_path -type f -name "depthquality*.xlsx" | head -n 1)
        python xlsx_to_csv.py $xlsx_path
        local csv_path="${xlsx_path%.xlsx}.csv"
    fi

    # Navigate to the build directory and execute the program
    cd build
    if [[ "$global_flag" == "true" ]]; then
        ./kbd -f "$file_path" -c "$csv_path" -t "$output_path" -g 2>&1 | tee -a "$log_file"
    else
        ./kbd -f "$file_path" -c "$csv_path" -t "$output_path" 2>&1 | tee -a "$log_file"
    fi
    cd - > /dev/null
}

if [[ "$mode" == "multiple" ]]; then
    # Process all directories
    for camera_type in $(ls -d $root_dir/*/); do
        log "Processing directory $camera_type with global flag: $use_global"
        process_directory "$camera_type" "$use_global"
    done
else
    # Process only the specified directory
    specific_dir="/Users/williamwei/Codes/KBD/data/N09ASH24DH0050"
    log "Processing single directory $specific_dir with global flag: $use_global"
    process_directory "$specific_dir" "$use_global"
fi
