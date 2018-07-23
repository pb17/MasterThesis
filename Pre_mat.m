clear 
clc
close all 
cd 'C:\Users\Samsung\Desktop\Pre Processed\DRIVE\training\images'
DRIVEFolder = dir('*.tif');
numFiles = length(DRIVEFolder);
DRIVE = cell(1,numFiles);
for k = 1:numFiles
    DRIVE{k} = imread(DRIVEFolder(k).name);
end
for k = 1:numFiles
%     subplot(4,5,k);
 %   figure;
 %   imshow(DRIVE{k});
end
% Pré Processamento 1
DRIVEG = cell(1,numFiles);
% STAREG = cell(1,numFiles2);
for k = 1:numFiles
   DRIVEG{k}=(DRIVE{k}(:,:,2)); 
   normalizedImage{k}= uint8(255*mat2gray(DRIVEG{k}));
   hist{k}= histeq(DRIVEG{k});
   G = fspecial('gaussian',[5 5],2);
   Ig{k} = imfilter(hist{k},G,'same');
   CLAE{k}=adapthisteq(normalizedImage{k},'NumTiles',[8 8],'ClipLimit',0.0005);
   gammaCorrectionD{k}= imadjust(CLAE{k},[],[],1.2);
   FinalDrive{k}=gammaCorrectionD{k};
end
figure(1)
imshow((DRIVE{1}(:,:,2)))
figure(2)
imshow(normalizedImage{1})
figure(3)
imshow(hist{1})
figure(4)
imshow(FinalDrive{1})