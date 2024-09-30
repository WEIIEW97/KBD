function  test_depth_transform()
%TEST_DEPTH_TRANSFORM Summary of this function goes here
%   Detailed explanation goes here
h=480;
w=640;

after_path = "D:\william\data\KBD\Z06FLAZG24GN0347\image_data\1800_Z06FLAZG24GN0347_2024_09_30_15_24_23\DEPTH\raw\Depth-2024-09-30-15-24-23-872-2-002048-1727681063349692.raw";
% before_path = "D:\william\data\KBD\fuck\first\image_data\500_N9LAZG24GN0130_2024_08_09_14_01_39\DEPTH\raw\Depth-2024-08-09-14-01-40-056-1-000556-41179234.raw";

% before = load_raw(before_path, h, w);
after = load_raw(after_path, h, w);

% d = delta_diff(before, after);

% Display the images
figure(1);
imagesc(after, [1600, 2000]);
colormap('default'); % You can choose any appropriate colormap
colorbar;
title('Before');

figure(2);
imagesc(after, [400, 600]);
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
