
clear
clc
close all

RGBImage = imread('Eozynofil.jpeg');

average = fspecial('average',2);
avIm = RGBImage;
minIm = RGBImage;
maxIm = RGBImage;

for i = 1:5
    minIm = imerode(minIm, true(2));
    avIm = imfilter(avIm,average);
    maxIm = imdilate(maxIm, true(2));
end

figure;
subplot(2,2,1)
imshow(RGBImage)
title('Oryginalne zdjęcie')

subplot(2,2,2)
imshow(avIm)
title('Filtr uśredniający')

subplot(2,2,3)
imshow(minIm)
title('Filt minimalizujący')

subplot(2,2,4)
imshow(maxIm)
title('Filtr maksymalizujący')