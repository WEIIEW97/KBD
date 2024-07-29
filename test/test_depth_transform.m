function  test_depth_transform()
%TEST_DEPTH_TRANSFORM Summary of this function goes here
%   Detailed explanation goes here
h=480;
w=640;

after_path = "D:\william\data\KBD\0723\N09ALC247H0116\image_data_transformed_linear_local_scale1.6\1000_N09ALC247H0116_2024_07_23_20_44_30\DEPTH\raw\Depth-2024-07-23-20-44-31-022-1-005654-1721738670329147.raw";
before_path = "D:\william\data\KBD\0723\N09ALC247H0116\image_data\1000_N09ALC247H0116_2024_07_23_20_44_30\DEPTH\raw\Depth-2024-07-23-20-44-31-022-1-005654-1721738670329147.raw";

before = load_raw(before_path, h, w);
after = load_raw(after_path, h, w);
% Display the images
figure(1);
imagesc(before, [800, 1200]);
colormap('default'); % You can choose any appropriate colormap
colorbar;
title('Before');

figure(2);
imagesc(after, [800, 1200]);
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