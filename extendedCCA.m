function [ecca_r1,ecca_r2,itcca_r,cca_r]=extendedCCA(X,Y,Z,num_of_r)
% X: real time multichannel signal
% Y: sine-cosine reference signal
% Z: individual SSVEP template

[Wx1,Wy1,cr1]=canoncorr(X',Y');
cca_r=cr1(1);
[Wx2,Wy2,cr2]=canoncorr(X',Z');
itcca_r=cr2(1);
[Wx3,Wy3,cr3]=canoncorr(Z',Y');
cr1=corrcoef((Wx1(:,1)'*X)',(Wy1(:,1)'*Y)');
cr2=corrcoef((Wx2(:,1)'*X)',(Wx2(:,1)'*Z)');
cr3=corrcoef((Wx1(:,1)'*X)',(Wx1(:,1)'*Z)');
cr4=corrcoef((Wx3(:,1)'*X)',(Wx3(:,1)'*Z)');
cr5=corrcoef((Wx2(:,1)'*Z)',(Wx3(:,1)'*Z)');
r_vector(1)=cr1(1,2);
r_vector(2)=cr2(1,2);
r_vector(3)=cr3(1,2);
r_vector(4)=cr4(1,2);
r_vector(5)=cr5(1,2);

r_sign=sign(r_vector);
ecca_r1=r_vector(1:num_of_r);
ecca_r2=r_sign(1:num_of_r)*(r_vector(1:num_of_r)).^2';

% ecca_r1=[cr1 cr2 cr3 cr4];
% ecca_r2=sign(cr1(1,2))*cr1(1,2)^2+sign(cr2(1,2))*cr2(1,2)^2+sign(cr3(1,2))*cr3(1,2)^2+sign(cr4(1,2))*cr4(1,2)^2;