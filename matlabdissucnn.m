Input = imread('ISIC_0000036.jpg');
SegmentedInput = imread('ISIC_0000036_Segmentation.png');

Igray = rgb2gray(Input);
Igray = padarray(Igray,[4,4]);

[rows, columns] = size(Igray);

J = zeros(rows,columns,'uint8');

%median filter
for row = 1:rows
    for col = 1:columns
        if (row <= 2) || (col <= 2) || col >= columns - 2 || row >= rows -2
            J(row, col) = 0;
        else
            mask = zeros(5, 5);
            for x = 1:5
                for y = 1:5
                    newx = x - 3;
                    newy = y - 3;
                    mask (x, y) = Igray(row - newx, col - newy);
                end
            end
            
            S = mask(mask ~= 0);
            S = sort(S, 'descend');
            A = S(ceil(end/2), :);
            J(row, col) = A;
        end
    end
end

J = J(4+1:end-4,4+1:end-4); % unpad

figure
imshowpair(Input, J, 'montage')
title('Original Image')

mask = zeros(size(J));
mask(300:end-300,300:end-300) = 1;

%figure
%imshow(mask)
%title('Initial Contour Location')

bw = activecontour(J,mask,1000);

se = strel('square',5);
border = imdilate(bw, se);
border = border - bw;

figure
imshow(bw)
title('Segmented Image')

bw2 = imdilate(bw, se);

Y = imbinarize(SegmentedInput);
Y = imfill(Y, 'holes');

similarity = jaccard(bw2, Y);

yolo = imfuse(bw2, Y);
title(['Jaccard Index = ' num2str(similarity)])
%saveas(yolo,'C:\Users\Harry\Desktop\Images\Jaccard.png');

B = imoverlay(Input,border);

figure
imshow(B)
title('Border overlaid on initial image')

bw = bwareafilt(bw,1);

stats = regionprops(bw,'Eccentricity', 'Extent', 'Centroid','Orientation', 'BoundingBox');
angle = -stats.Orientation;
rotatedImage = imrotate(bw, angle, 'crop');

imshow(rotatedImage);

val1 = stats.BoundingBox(1);
val2 = stats.BoundingBox(2);
val3 = stats.BoundingBox(3);
val4 = stats.BoundingBox(4);

box = [val1, val2, val3, val4];    
cropped = imcrop(Input,box);
figure, imshow(cropped), title('Lesion area');

croppedgray = rgb2gray(cropped);

bw2 = imbinarize(croppedgray);
bw2 = imcomplement(bw2);
%imshow(bw2);
bw2 = bwareafilt(bw2,1);

stats = regionprops(bw2,'Eccentricity', 'Extent', 'Centroid','Orientation', 'BoundingBox', 'Area');

[rows, columns, numberOfColorChannels] = size(cropped);

middlex = columns/2;
middley = rows/2;

xCentroid = stats.Centroid(1);
yCentroid = stats.Centroid(2);

deltax = middlex - xCentroid;
deltay = middley - yCentroid;

distancex = xCentroid + deltax; 
distancey = yCentroid + deltay; 

%croppedmiddle = insertMarker(cropped,[xCentroid yCentroid], 'color', 'magenta' ,'size', 10);
%croppedmiddle = insertMarker(croppedmiddle,[middlex middley], 'color', 'white','size', 10);

ir = imresize(cropped,[32 32]);

load disscnn;
newoutput = classify(disscnn, ir);
%newoutput

% 3 x 2 grid of images, in slot x plot image y
subplot(2,2,1), imshow(Input);
title('Original Image')  

subplot(2,2,2), imshow(Y);
title('Shape of object detected')  

subplot(2,2,3), imshow(B);
title('Edges ploted onto original image')  

subplot(2,2,4), imshow(cropped);
title('Crop non important areas') 

%subplot(2,2,1), imshow(croppedmiddle);
%title('Crop non important areas')

%Split into RGB Channels
%Red = cropped(:,:,1);
%Green = cropped(:,:,2);
%Blue = cropped(:,:,3);
%Get histValues for each channel

%histo = figure;
%imhist(Red);
%saveas(histo, 'C:\Users\Harry\Desktop\Images\redhistogram.jpg')

%[yGreen, x] = imhist(Green);
%[yBlue, x] = imhist(Blue);
%Plot them together in one plot

imwrite(cropped,'C:\Users\Harry\Desktop\Images\cropped.jpg');
imwrite(Y,'C:\Users\Harry\Desktop\Images\Y.jpg');
imwrite(B,'C:\Users\Harry\Desktop\Images\B.jpg');
imwrite(Input,'C:\Users\Harry\Desktop\Images\I.jpg');
imwrite(yolo,'C:\Users\Harry\Desktop\Images\Jaccard.jpg');
%imwrite(croppedmiddle,'C:\Users\Harry\Desktop\Images\Irregular.png');

h = cellstr(newoutput);
h = string(h);
%msgbox(h);

%rootFolder = 'C:\Users\Harry\Desktop\Images\Test';
%imds_test = imageDatastore(fullfile(rootFolder, categories), ...
 %   'LabelSource', 'foldernames');

%labels = classify(net, imds_test);

% This could take a while if you are not using a GPU
%confMat = confusionmat(imds_test.Labels, labels);
%confMat = confMat./sum(confMat,2);


fid=fopen('C:\Users\Harry\Desktop\Images\results.txt', 'w+');
fprintf(fid, 'Border Accuracy -  %f \n', num2str(similarity));
fprintf(fid, 'CNN Prediction - %s', h);
fclose(fid);