% clear 
% clc
% close all 
% cd 'C:\Users\Samsung\Desktop\Pre Processed\STARE_1\Traning\image\Pre pro'
% DRIVEFolder = dir('*.tif');
% numFiles = length(DRIVEFolder);
% DRIVE = cell(1,numFiles);
% for k = 1:numFiles
%     DRIVE{k} = imread(DRIVEFolder(k).name);
% end
% for k = 1:numFiles
% %     subplot(4,5,k);
%  %   figure;
%  %   imshow(DRIVE{k});
% end
% % Pré Processamento 1
% DRIVEG = cell(1,numFiles);
% % STAREG = cell(1,numFiles2);
% for k = 1:numFiles
%    DRIVEG{k}=(DRIVE{k}(:,:,2)); 
%    normalizedImage{k}= uint8(255*mat2gray(DRIVEG{k}));
%    CLAE{k}=adapthisteq(normalizedImage{k},'NumTiles',[8 8],'ClipLimit',0.005);
%    gammaCorrectionD{k}= imadjust(CLAE{k},[],[],0.8);
% end
% for k = 1:1


cd 'C:\Users\Samsung\Desktop\Pre Processed\STARE_1\Test\images'
STAREFolder = dir('*.ppm');
numFiles2 = length(STAREFolder);
STARE = cell(1,numFiles2);
for k = 1:numFiles2
    STARE{k} = imread(STAREFolder(k).name);
end


for k = 1:numFiles2
  STAREG{k}=(STARE{k}(:,:,2));
  normalizedImage{k}= uint8(255*mat2gray(STAREG{k}));
  CLAE{k}=adapthisteq(normalizedImage{k},'NumTiles',[8 8],'ClipLimit',0.005);
  gammaCorrection{k}= imadjust(CLAE{k},[],[],1.2);
end


 for k = 1:1
% %     subplot(4,5,k);
    figure;
    imshow(gammaCorrection{10});
   cd 'C:\Users\Samsung\Desktop\Pre Processed\STARE_1\Test\Pre_pro'
   % imwrite(gammaCorrection{10}, 'STARE_Pre_10.ppm','ppm')
 end






