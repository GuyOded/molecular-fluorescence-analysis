A=imread('1736340621623.jpg');
A1=im2double(A);
figure("Name", "image")
imagesc(A1(:,:,1))

figure("Name", "graph")
Av=A1(984,1286:2400,1);
x=linspace(0,10,length(Av));
plot(x,Av)

Avl=log(Av);
plot(x,Avl)
xlabel('x[cm]')
ylabel('Power [AU]')
grid minor

%Intenstiy = transpose(Av);
%X = transpose(x);
figure("Name", "image 2")
A1 = im2double(A)
imagesc(A1(:,:,1))
imagesc(A1(:,:,2))
imagesc(A1(:,:,3))
imagesc(A1(:,:,1))
imagesc(A1(:,:,1))
imagesc(A1(:,:,2))
imagesc(A1(:,:,3))
imagesc(A1(:,:,1))