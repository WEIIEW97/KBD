function  test_depth_transform()
%TEST_DEPTH_TRANSFORM Summary of this function goes here
%   Detailed explanation goes here
h=480;
w=640;

before_path = "/Users/williamwei/Data/KBD/N09ALC247H0070/image_data/1500_N09ALC247H0070_2024_07_21_23_12_08/DEPTH/raw/Depth-2024-07-21-23-12-08-866-1-001048-1721574728236050.raw";
after_path = "/Users/williamwei/Data/KBD/N09ALC247H0070/image_data_l/1500_N09ALC247H0070_2024_07_21_23_12_08/DEPTH/raw/Depth-2024-07-21-23-12-08-866-1-001048-1721574728236050.raw";

before = load_raw(before_path, h, w);
after = load_raw(after_path, h, w);
% Display the images
figure(1);
imagesc(before, [1200, 1800]);
colormap('default'); % You can choose any appropriate colormap
colorbar;
title('Before');

figure(2);
imagesc(after, [1200, 1800]);
colormap('default');
colorbar;
title('After');

end

function data = load_raw(filename, h, w)
    % Open the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Read the file data
    data = fread(fid, [w, h], 'uint16=>uint16');
    data = data'; % Transpose to match the (h, w) format
    
    % Close the file
    fclose(fid);
end
