%% Double null configuration

% Define magnetic field and take derivatives

clear
clc

syms x y %xmag1 y0 x1 y1 x2 y2 I0 s

Lx=600;
Ly=800;
R =700;
nx=244;
ny=324;

dx=Lx/(nx-4);
dy=Ly/(ny-4);
xv=-3/2*dx:dx:(Lx+3*dx/2);
yv=-3/2*dy:dy:(Ly+3*dy/2);
yn=-1/2*dy:dy:(Ly+5*dy/2);
[Xv,Yv]=meshgrid(xv,yv);
[Xn,Yn]=meshgrid(xv,yn);

%%
left_corr = 0;
Yxpt_low = 0.18*Ly; %Yxpt_low
Yxpt_up  = Ly-Yxpt_low; %Yxpt1 => symmetric by X axis
Xxpt = 0.6*Lx - left_corr;  % Same x-coord for both null points

xmag1  = 0.6 * Lx- left_corr;   % X-coord. of the main plasma current
xmag2 = Lx/2 -60- left_corr;    % Lower divertor current
xmag3 = xmag2;       % Upper divertor current

% Right shaping for PT
xmag4 = Lx + 1.2 * Lx - 25- left_corr; % Left lower shaping current
xmag5 = xmag4;              % Left upper shaping current


ymag1 = 0.5*Ly ;     % Y-coord. of the main plasma current
ymag2 = ymag1 - sqrt(3)*(ymag1-Yxpt_low);
ymag3 = ymag1 + sqrt(3)*(ymag1-Yxpt_low);
ymag4 = ymag1 - sqrt(3)*(ymag1-Yxpt_low); % Upper shaping current
ymag5 = ymag1 + sqrt(3)*(ymag1-Yxpt_low); % Lower shaping current

c1 = 1.57;      %DOFs
c2 = 1.55;
c3 = 1.55;
c4 = 0.5;
c5 = 0.5;
I0 = 2000*Ly/400*Lx/300;
s = 60*Ly/400; % instead of s

%% Symbolic Expressions for magetic fields

F(x,y) = (1-exp(-((x-xmag1)^2+(y-ymag1)^2)/s^2));

bx0(x,y) = I0*F(x,y)*(y-ymag1)/((x-xmag1)^2+(y-ymag1)^2);
by0(x,y) = -I0*F(x,y)*(x-xmag1)/((x-xmag1)^2+(y-ymag1)^2);

bx1(x,y) = I0*(y-ymag2)/((x-xmag2)^2+(y-ymag2)^2);
by1(x,y) = -I0*(x-xmag2)/((x-xmag2)^2+(y-ymag2)^2);

bx2(x,y) = I0*(y-ymag3)/((x-xmag3)^2+(y-ymag3)^2);
by2(x,y) = -I0*(x-xmag3)/((x-xmag3)^2+(y-ymag3)^2);

bx3(x,y) = I0*(y-ymag4)/((x-xmag4)^2+(y-ymag4)^2);
by3(x,y) = -I0*(x-xmag4)/((x-xmag4)^2+(y-ymag4)^2);

bx4(x,y) = I0*(y-ymag5)/((x-xmag5)^2+(y-ymag5)^2);
by4(x,y) = -I0*(x-xmag5)/((x-xmag5)^2+(y-ymag5)^2);

bx(x,y) = bx0 + bx1 + bx2 + bx3 + bx4;  % Total mag fields
by(x,y) = by0 + by1 + by2 + by3 + by4;

dpsidx(x,y) = -by(x,y);     % psi? 
dpsidy(x,y) = bx(x,y);

d2psidx2(x,y) = diff(dpsidx,x);
d2psidy2(x,y) = diff(dpsidy,y);
d2psidxy(x,y) = diff(dpsidx,y);

%% Fill arrays 

dpdx = matlabFunction(subs(dpsidx));
dpdy = matlabFunction(subs(dpsidy));
d2pdx2 = matlabFunction(subs(d2psidx2));
d2pdy2 = matlabFunction(subs(d2psidy2));
d2pdxy = matlabFunction(subs(d2psidxy));


dpsidx_n = dpdx(Xn,Yn);
dpsidx_v = dpdx(Xv,Yv);
dpsidy_n = dpdy(Xn,Yn);
dpsidy_v = dpdy(Xv,Yv);
d2psidx2_n = d2pdx2(Xn,Yn);
d2psidx2_v = d2pdx2(Xv,Yv);
d2psidy2_n = d2pdy2(Xn,Yn);
d2psidy2_v = d2pdy2(Xv,Yv);
d2psidxy_n = d2pdxy(Xn,Yn);
d2psidxy_v = d2pdxy(Xv,Yv);


Psi0 = I0/2*c1*(log((Xv-xmag1).^2+(Yv-ymag1).^2));%+expint(((Xv-xmag1).^2+(Yv-ymag1).^2)/s^2));
Psi1 = I0/2*c2*(log((Xv-xmag2).^2+(Yv-ymag2).^2));
Psi2 = I0/2*c3*(log((Xv-xmag3).^2+(Yv-ymag3).^2));
Psi3 = I0/2*c4*(log((Xv-xmag4).^2+(Yv-ymag4).^2));
Psi4 = I0/2*c5*(log((Xv-xmag5).^2+(Yv-ymag5).^2));
Psi = Psi0 + Psi1 + Psi2 + Psi3 + Psi4;
%%
% Plot magnetic field

x=xv;
y=yv;
iYxpt=find(abs(y-Yxpt_low)<=dy/2,1); % index of nullpoint in Y axis mesh

Xs = [xmag1 xmag2 xmag3 xmag4 xmag5]; % Coordinates of auxiliary currents
Ys = [ymag1 ymag2 ymag3 ymag4 ymag5];

[X,Y]=meshgrid(x,y);
figure;
contour(X,Y,Psi,Psi(end/2,end/2):Psi(end/2,end/2)/200:Psi(end/2,end-5),'k')
hold on
contour(X,Y,Psi,'r','LevelList',Psi(iYxpt,end/2));
for i = 1:5
    plot(Xs(i), Ys(i), 'o')
end
% contourf(X,Y,Psi,40,'EdgeColor','None')
xlabel('$R/\rho_{s0}$','Interpreter','latex')
ylabel('$Z/\rho_{s0}$','Interpreter','latex')
title('$\Psi$','Interpreter','latex')
axis equal

% figure;
% contourf(X,Y,dpsidy_v,50,'EdgeColor','none')
% axis equal
% colorbar
% title('$B_x$','Interpreter','latex')
% xlabel('R/\rho_{s0}')
% ylabel('Z/\rho_{s0}')
% 
% figure;
% contourf(X,Y,-dpsidx_v,50,'EdgeColor','none')
% axis equal
% colorbar
% title('$B_y$','Interpreter','latex')
% xlabel('R/\rho_{s0}')
% ylabel('Z/\rho_{s0}')
% 
% figure;
% contourf(X,Y,sqrt(dpsidx_v.^2+dpsidy_v.^2)/R,50,'EdgeColor','none')
% axis equal
% colorbar
% title('$B_{pol}$','Interpreter','latex')
% xlabel('R/\rho_{s0}')
% ylabel('Z/\rho_{s0}')

%% Compute local safety factor and magnetic shear

x=xv;
y=yv;
iYxpt=find(abs(y-Yxpt_low)<=dy/2,1);
iXxpt=find(abs(x-Xxpt)<=dy/2,1);

[X, Y] =meshgrid(x,y);

% Use symmetry to find yc (center coordinate)
yzeros = InterX([y;0*y],[y;dpsidy_v(:,end/2)']);
yc=yzeros(1,2);
iyc=find(abs(y-yc)<=dy/2,1);

% Find a
hh=figure(100);
XY=contour(X,Y,Psi,'LevelList',Psi(iYxpt,iXxpt));
XX=XY(1,2:end);
YY=XY(2,2:end);
close(hh)
r=linspace(xmag1,x(end),100);
r = InterX([r;yc*ones(size(r))], [XX;YY]);

% (LIM) 
%line3 = [r;yc*ones(size(r))]
%line4 = [XX;YY]


a = r(1,1) - xmag1;
ia = find(abs(x-xmag1-a)<=dx/2,1);
ix0 = find(abs(x-xmag1)<=dx/2,1);

r=x(ix0:ia)-xmag1;
q=NaN(1,length(r));
%Local q profile
for i=ix0:ia
    idx = i - ix0 +1;
    q(idx)=r(idx)/dpsidx_v(iyc,i);
end

%Shear
s=NaN(size(q));
dr=r(2)-r(1);
for i=2:length(q)-1
    s(i)=r(i)/q(i)*(q(i+1)-q(i-1))/(2*dr);
end    

figure;
plot(r/a,q)
xlabel('r/a')
ylabel('q(r)')
title('Local q profile at $\theta$ = 0','Interpreter','latex')

figure;
plot(r/a,s)
xlabel('r [\rho_s]')
ylabel('s(r)')
title('Local shear at $\theta$ = 0','Interpreter','latex')

%% Compute average safety factor

[X,Y]=meshgrid(x,y);
nk=200;
qpsi=zeros(1,nk);
rpsi=zeros(1,nk);
psi0=Psi(round((iyc+iYxpt)/2),iXxpt);
psi1=Psi(iYxpt+5,iXxpt);
psilev=linspace(psi0,psi1,nk);
for k=1:nk
    hh=figure(100);
    XY=contour(X,Y,Psi,'LevelList',psilev(k));
    axis equal
    XX=XY(1,2:end);
    YY=XY(2,2:end);
    qpsi(k)=0;
    rpsi(k)=(psilev(k)-Psi(iyc,iXxpt))/(Psi(iYxpt,iXxpt)-Psi(iyc,iXxpt));
    % Select only the points which belong to the closed field line
    Nr=0;
    rdist = [];
    dpsidr = [];
    qloc = [];
    for i=1:length(XX)
        if(sqrt((XX(i)-xmag1)^2+(YY(i)-yc)^2)<abs(yc-Yxpt_low))
            Nr=Nr+1;
            iX=find(abs(XX(i)-x)<=dx/2,1);
            iY=find(abs(YY(i)-y)<=dy/2,1);
            rdist(Nr)=sqrt((XX(i)-xmag1)^2+(YY(i)-yc)^2);
            dpsidr(Nr)=sqrt(dpsidx_v(iY,iX)^2+dpsidy_v(iY,iX)^2);
            qloc(Nr)=rdist(Nr)/dpsidr(Nr);
        end
    end
    % Compute q for each surface
    qpsi(k)=mean(qloc);
    
    pause(0.01)
end

close(hh)
figure;
plot(rpsi,qpsi)
xlabel('(\Psi-\Psi_0)/(\Psi_{LCFS}-\Psi_0)')
ylabel('q(\Psi)')
title('Averaged q profile')

%% Save equilibrium

!rm equilibrium
filename = 'equilibrium.mat';
nghost=2;

% Saves one field after another in a text file
fid = fopen(filename,'a');
fprintf(fid,'%s\n','&MAG_FIELD');
fclose(fid);
save_2darray_gbs(filename,'psi_eq',Psi,nghost);
save_2darray_gbs(filename,'dpsidx_v',dpsidx_v,nghost);
save_2darray_gbs(filename,'dpsidy_v',dpsidy_v,nghost);
save_2darray_gbs(filename,'d2psidx2_v',d2psidx2_v,nghost);
save_2darray_gbs(filename,'d2psidy2_v',d2psidy2_v,nghost);
save_2darray_gbs(filename,'d2psidxdy_v',d2psidxy_v,nghost);
save_2darray_gbs(filename,'dpsidx_n',dpsidx_n,nghost);
save_2darray_gbs(filename,'dpsidy_n',dpsidy_n,nghost);
save_2darray_gbs(filename,'d2psidx2_n',d2psidx2_n,nghost);
save_2darray_gbs(filename,'d2psidy2_n',d2psidy2_n,nghost);
save_2darray_gbs(filename,'d2psidxdy_n',d2psidxy_n,nghost);
save_real_gbs(filename,'xmag1',xmag1)
save_real_gbs(filename,'y0_source',yc)
save_real_gbs(filename,'Yxpt_low',Yxpt_low)
save_real_gbs(filename,'Yxpt_up',Yxpt_up)
fid = fopen(filename,'a');
fprintf(fid,'%s\n','/');
fclose(fid);
%%
%LIM
h5create('qprofile.h5', '/qpsi', [1,200])
h5write('qprofile.h5', '/qpsi', qpsi)
h5create('qprofile.h5', '/rpsi', [1,200])
h5write('qprofile.h5', '/rpsi', rpsi)
h5create('qprofile.h5', '/q', [1,68])
h5write('qprofile.h5', '/q', q)
h5create('qprofile.h5', '/s', [1,68])
h5write('qprofile.h5', '/s', s)
h5create('qprofile.h5', '/r_a', [1,68])
h5write('qprofile.h5', '/r_a', r/a)
clear('filename')
%%
h5create('DN_equilibrium.h5', '/psi_eq', [324,244])
h5write('DN_equilibrium.h5', '/psi_eq', Psi)
h5create('DN_equilibrium.h5', '/dpsidx_v', [324,244])
h5write('DN_equilibrium.h5', '/dpsidx_v', dpsidx_v)
h5create('DN_equilibrium.h5', '/dpsidy_v', [324,244])
h5write('DN_equilibrium.h5', '/dpsidy_v', dpsidy_v)
h5create('DN_equilibrium.h5', '/d2psidx2_v', [324,244])
h5write('DN_equilibrium.h5', '/d2psidx2_v', d2psidx2_v)
h5create('DN_equilibrium.h5', '/d2psidy2_v', [324,244])
h5write('DN_equilibrium.h5', '/d2psidy2_v', d2psidy2_v)
h5create('DN_equilibrium.h5', '/xmag1', [1,1])
h5write('DN_equilibrium.h5', '/xmag1', xmag1)
h5create('DN_equilibrium.h5', '/y0_source', [1,1])
h5write('DN_equilibrium.h5', '/y0_source', yc)
h5create('DN_equilibrium.h5', '/Yxpt_low', [1,1])
h5write('DN_equilibrium.h5', '/Yxpt_low', Yxpt_low)
h5create('DN_equilibrium.h5', '/Yxpt_up', [1,1])
h5write('DN_equilibrium.h5', '/Yxpt_up', Yxpt_up)
clear('filename')