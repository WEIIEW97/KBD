function  test_depth_transform()
%TEST_DEPTH_TRANSFORM Summary of this function goes here
%   Detailed explanation goes here
h=480;
w=640;

after_path = "/Users/williamwei/Data/KBD/20240803/N9LAZG24GN0197/image_data_transformed_linear_local_scale1.6/1200_N9LAZG24GN0197_2024_08_02_21_18_46/DEPTH/raw/Depth-2024-08-02-21-18-46-200-1-005435-1722604725641614.raw";
before_path = "/Users/williamwei/Data/KBD/20240803/N9LAZG24GN0197/image_data/1200_N9LAZG24GN0197_2024_08_02_21_18_46/DEPTH/raw/Depth-2024-08-02-21-18-46-200-1-005435-1722604725641614.raw";

before = load_raw(before_path, h, w);
after = load_raw(after_path, h, w);

d = delta_diff(before, after);

% Display the images
figure(1);
imagesc(before, [1000, 1300]);
colormap('default'); % You can choose any appropriate colormap
colorbar;
title('Before');

figure(2);
imagesc(after, [1000, 1300]);
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

function delta = delta_diff(a, b)
    diff = a - b;
    delta = sum(diff,"all");
end
