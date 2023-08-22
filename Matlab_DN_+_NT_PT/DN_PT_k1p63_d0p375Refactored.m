%% Double null configuration

% Define magnetic field and take derivatives

clear 
clc

%Inputs
kappaTarget = 1.63;
deltaTarget = -0.6;

Lx=600;
Ly=800;
R =700;
nx=244;
ny=324;

dx=Lx/(nx-4);
dy=Ly/(ny-4);
xv=-3/2*dx:dx:(Lx+3*dx/2);
yv=-3/2*dy:dy:(Ly+3*dy/2);
[Xv,Yv]=meshgrid(xv,yv);

I0 = 2000*Ly/400*Lx/300;
s = 60*Ly/400; % instead of s


%% Initial elliptical guess
paramX = 607.5;

Yxpt_low = 0.18*Ly; %Yxpt_low
Yxpt_up  = Ly-Yxpt_low; %Yxpt1 => symmetric by X axis

x0  = 0.6*Lx;   % X-coord. of the main plasma current
x1 = Lx/2 -60;    % Lower divertor current
x2 = x1;       % Upper divertor current

% Right shaping for PT
%x4 = Lx + 1.2 * Lx - 25- left_corr; % Left lower shaping current

x3 = 2* x0 - x1;   % Symmetric for elliptical initial guess
x4 = x3;              % Left upper shaping current


y0 = 0.5*Ly ;     % Y-coord. of the main plasma current
y1 = y0 - sqrt(3)*(y0-Yxpt_low);
y2 = y0 + sqrt(3)*(y0-Yxpt_low);
y3 = y0 - sqrt(3)*(y0-Yxpt_low); % Upper shaping current
y4 = y0 + sqrt(3)*(y0-Yxpt_low); % Lower shaping current




c0 = 1;      %DOFs
caux = 0.75;
c1 = caux;
c2 = caux;
c3 = caux;
c4 = caux;
plotMagField(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s, xv, yv)
%%
% Choice of ideal caux to reach kappaTarget without triangularity
% Caux stands for auxiliary current

disp("Optimizing elongation...")


COp = optimizeKappa(kappaTarget, caux, c0, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s);

c0 = COp(1);
caux = COp(2);


disp("caux reached after optimization : ");
disp(caux)

c1 = caux;
c2 = caux;
c3 = caux;
c4 = caux;

[~,kappa, Rmin, Rmax] = computeSpecs(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s);

disp("kappa reached after elongation optimization : ");
disp(kappa);
disp("Rmin Rmax : ")
disp(Rmin)
disp(Rmax)

plotMagField(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s, xv, yv);


%% Study of parameters
% 
% 
% cl = caux;  % Current on left divertors (symmetric)
% cr = caux;  % Current on right divertors
% xl = x1; % x coord of left divertors (symmetric)
% 
% % Study of cl
% 
% cl_ = linspace(cl, cl*2, 20 );
% deltas_cl = [];
% 
% for i = 1:length(cl_)
%     c1 = cl_(i);
%     c2 = cl_(i);
%     c3 = cr;
%     c4 = cr;
%     [delta, kappa, Rmin, Rmax] = computeSpecs(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
%     x2, y2, x3, y3, x4, y4, I0, xv, yv,s);
%     deltas_cl = [deltas_cl delta];
%     %plotMagField(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
%     %x2, y2, x3, y3, x4, y4, I0, s, xv, yv)
% end

%% Optimize for NT

cl = caux;  % Current on left divertors (symmetric)
cr = caux;  % Current on right divertors
xl = x1; % x coord of left divertors (symmetric)

X = [xl, cl, cr, c0];    % Initial guess

disp("Optimizing triangularity...")

XOp = optimizeDelta(deltaTarget, kappaTarget, X, x0, y0, y1, ...
    y2, x3, y3, x4, y4, I0, xv, yv,s);


x1 = XOp(1);
x2 = x1;
c1 = XOp(2);
c2 = c1;
c3 = XOp(3);
c4 = c3;
c0 = XOp(4);

disp("c0 = ")
disp(c0)
disp("c1 = ")
disp(c1)
disp("c2 = ")
disp(c2)
disp("c3 = ")
disp(c3)
disp("c4 = ")
disp(c4)
disp("x0 = ")
disp(x0)
disp("x1 = ")
disp(x1)
disp("x2 = ")
disp(x2)
disp("x3 = ")
disp(x3)
disp("x4 = ")
disp(x4)

[delta, kappa, Rmin, Rmax] = computeSpecs(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s);


disp("Delta obtained after optimization : ")
disp(delta)
disp("Kappa obtained after delta-optimization : ")
disp(kappa)

disp("Rmin Rmax")
disp(Rmin)
disp(Rmax)

% Plot the field

plotMagField(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s, xv, yv); 


%% Compute local safety factor and magnetic shear
% 
% x=xv;
% y=yv;
% 
% [X, Y] =meshgrid(x,y);
% 
% % Use symmetry to find yc (center coordinate)
% yzeros = InterX([y;0*y],[y;dpsidy_v(:,end/2)']);
% yc=yzeros(1,2);
% iyc=find(abs(y-yc)<=dy/2,1);
% 
% % Find a
% hh=figure(100);
% XY=contour(X,Y,Psi,'LevelList',Psi(iYxptlow,iXxpt));
% XX=XY(1,2:end);
% YY=XY(2,2:end);
% close(hh)
% r=linspace(x0,x(end),100);
% r = InterX([r;yc*ones(size(r))], [XX;YY]);
% 
% % (LIM) 
% %line3 = [r;yc*ones(size(r))]
% %line4 = [XX;YY]
% 
% 
% a = r(1,1) - x0;
% ia = find(abs(x-x0-a)<=dx/2,1);
% ix0 = find(abs(x-x0)<=dx/2,1);
% 
% r=x(ix0:ia)-x0;
% q=NaN(1,length(r));
% %Local q profile
% for i=ix0:ia
%     idx = i - ix0 +1;
%     q(idx)=r(idx)/dpsidx_v(iyc,i);
% end
% 
% %Shear
% s=NaN(size(q));
% dr=r(2)-r(1);
% for i=2:length(q)-1
%     s(i)=r(i)/q(i)*(q(i+1)-q(i-1))/(2*dr);
% end    
% 
% figure;
% plot(r/a,q)
% xlabel('r/a')
% ylabel('q(r)')
% title('Local q profile at $\theta$ = 0','Interpreter','latex')
% 
% figure;
% plot(r/a,s)
% xlabel('r [\rho_s]')
% ylabel('s(r)')
% title('Local shear at $\theta$ = 0','Interpreter','latex')
% 
% %% Compute average safety factor
% 
% [X,Y]=meshgrid(x,y);
% nk=200;
% qpsi=zeros(1,nk);
% rpsi=zeros(1,nk);
% psi0=Psi(round((iyc+iYxptlow)/2),iXxpt);
% psi1=Psi(iYxptlow+5,iXxpt);
% psilev=linspace(psi0,psi1,nk);
% for k=1:nk
%     hh=figure(100);
%     XY=contour(X,Y,Psi,'LevelList',psilev(k));
%     axis equal
%     XX=XY(1,2:end);
%     YY=XY(2,2:end);
%     qpsi(k)=0;
%     rpsi(k)=(psilev(k)-Psi(iyc,iXxpt))/(Psi(iYxptlow,iXxpt)-Psi(iyc,iXxpt));
%     % Select only the points which belong to the closed field line
%     Nr=0;
%     rdist = [];
%     dpsidr = [];
%     qloc = [];
%     for i=1:length(XX)
%         if(sqrt((XX(i)-x0)^2+(YY(i)-yc)^2)<abs(yc-Yxpt_low))
%             Nr=Nr+1;
%             iX=find(abs(XX(i)-x)<=dx/2,1);
%             iY=find(abs(YY(i)-y)<=dy/2,1);
%             rdist(Nr)=sqrt((XX(i)-x0)^2+(YY(i)-yc)^2);
%             dpsidr(Nr)=sqrt(dpsidx_v(iY,iX)^2+dpsidy_v(iY,iX)^2);
%             qloc(Nr)=rdist(Nr)/dpsidr(Nr);
%         end
%     end
%     % Compute q for each surface
%     qpsi(k)=mean(qloc);
%     
%     pause(0.01)
% end
% 
% close(hh)
% figure;
% plot(rpsi,qpsi)
% xlabel('(\Psi-\Psi_0)/(\Psi_{LCFS}-\Psi_0)')
% ylabel('q(\Psi)')
% title('Averaged q profile')
% 
% %% Save equilibrium
% 
% !rm equilibrium
% filename = 'equilibrium.mat';
% nghost=2;
% 
% % Saves one field after another in a text file
% fid = fopen(filename,'a');
% fprintf(fid,'%s\n','&MAG_FIELD');
% fclose(fid);
% save_2darray_gbs(filename,'psi_eq',Psi,nghost);
% save_2darray_gbs(filename,'dpsidx_v',dpsidx_v,nghost);
% save_2darray_gbs(filename,'dpsidy_v',dpsidy_v,nghost);
% save_2darray_gbs(filename,'d2psidx2_v',d2psidx2_v,nghost);
% save_2darray_gbs(filename,'d2psidy2_v',d2psidy2_v,nghost);
% save_2darray_gbs(filename,'d2psidxdy_v',d2psidxy_v,nghost);
% save_2darray_gbs(filename,'dpsidx_n',dpsidx_n,nghost);
% save_2darray_gbs(filename,'dpsidy_n',dpsidy_n,nghost);
% save_2darray_gbs(filename,'d2psidx2_n',d2psidx2_n,nghost);
% save_2darray_gbs(filename,'d2psidy2_n',d2psidy2_n,nghost);
% save_2darray_gbs(filename,'d2psidxdy_n',d2psidxy_n,nghost);
% save_real_gbs(filename,'x1',x0)
% save_real_gbs(filename,'y0_source',yc)
% save_real_gbs(filename,'Yxpt_low',Yxpt_low)
% save_real_gbs(filename,'Yxpt_up',Yxpt_up)
% fid = fopen(filename,'a');
% fprintf(fid,'%s\n','/');
% fclose(fid);
% %%
% %LIM
% h5create('qprofile.h5', '/qpsi', [1,200])
% h5write('qprofile.h5', '/qpsi', qpsi)
% h5create('qprofile.h5', '/rpsi', [1,200])
% h5write('qprofile.h5', '/rpsi', rpsi)
% h5create('qprofile.h5', '/q', [1,68])
% h5write('qprofile.h5', '/q', q)
% h5create('qprofile.h5', '/s', [1,68])
% h5write('qprofile.h5', '/s', s)
% h5create('qprofile.h5', '/r_a', [1,68])
% h5write('qprofile.h5', '/r_a', r/a)
% clear('filename')
% %%
% h5create('DN_equilibrium.h5', '/psi_eq', [324,244])
% h5write('DN_equilibrium.h5', '/psi_eq', Psi)
% h5create('DN_equilibrium.h5', '/dpsidx_v', [324,244])
% h5write('DN_equilibrium.h5', '/dpsidx_v', dpsidx_v)
% h5create('DN_equilibrium.h5', '/dpsidy_v', [324,244])
% h5write('DN_equilibrium.h5', '/dpsidy_v', dpsidy_v)
% h5create('DN_equilibrium.h5', '/d2psidx2_v', [324,244])
% h5write('DN_equilibrium.h5', '/d2psidx2_v', d2psidx2_v)
% h5create('DN_equilibrium.h5', '/d2psidy2_v', [324,244])
% h5write('DN_equilibrium.h5', '/d2psidy2_v', d2psidy2_v)
% h5create('DN_equilibrium.h5', '/x1', [1,1])
% h5write('DN_equilibrium.h5', '/x1', x0)
% h5create('DN_equilibrium.h5', '/y0_source', [1,1])
% h5write('DN_equilibrium.h5', '/y0_source', yc)
% h5create('DN_equilibrium.h5', '/Yxpt_low', [1,1])
% h5write('DN_equilibrium.h5', '/Yxpt_low', Yxpt_low)
% h5create('DN_equilibrium.h5', '/Yxpt_up', [1,1])
% h5write('DN_equilibrium.h5', '/Yxpt_up', Yxpt_up)
% clear('filename')

%% Functions

function [iX, iY] = xptCoords(xv, yv, dpdx, dpdy)
% Returns indices of lower X point in mesh 


 [X, Y] = meshgrid(xv, yv);

 Bp = dpdx(X, Y).^2 + dpdy(X,Y).^2;
 [M, iY] = min(Bp);
 [~, iX] = min(M);
 
 iY = iY(iX);    
   
 if iY > size(yv,2)/2       % If upper Xpoint detected, switch to lower by sym
    iY = size(yv,2) - iY+1;
 end
 
end

% function kappa = computeKappa(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
%     x2, y2, x3, y3, x4, y4, I0, xv, yv,s)
% 
% % Computes elongation out of the variables of the problem
% 
% [dpdx, dpdy] = computeDpsidr(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
%     x2, y2, x3, y3, x4, y4, I0, s);
% Psi = computePsi(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
%     x2, y2, x3, y3, x4, y4, I0, xv, yv,s);
% 
% % Find xpoint coordinates
% 
% [iXxpt, iYxptlow] = xptCoords(xv, yv, dpdx, dpdy);
% iYxptup = size(yv, 2) - iYxptlow;
% 
% Yxptlow = yv(iYxptlow);
% Yxptup = yv(iYxptup);
% 
% [X,Y] = meshgrid(xv,yv);
% 
% 
% %Obtain coordinates of points on separatrix
% c = contour(X,Y,Psi,'r','LevelList',Psi(iXxpt,iYxptlow));
% sepX = [];
% sepY = [];
% inb = 1;
% 
% while inb < size(c, 2)
%     iend = inb + c(2, inb);
%     sepX = [sepX c(1, inb+1:iend)]; 
%     sepY = [sepY c(2, inb+1:iend)];
%     inb = iend+1;
% end
% 
% Rmin = min(sepX(:, iYxptlow:iYxptup));
% Rmax = max(sepX(:, iYxptlow:iYxptup));
% 
% kappa = (Yxptup-Yxptlow)/(Rmax-Rmin);
% 
% end

function [dpdx, dpdy] = computeDpsidr(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s)

% Computes the gradient of flux out of the variables of the problem

syms x y

F(x,y) = 1-exp(-((x-x0)^2+(y-y0)^2)/s^2)*((x-x0)^2+(y-y0)^2)/s^2;

bx0(x,y) = c0*I0*F(x,y)*(y-y0)/((x-x0)^2+(y-y0)^2) ;
by0(x,y) = -c0*I0*F(x,y)*(x-x0)/((x-x0)^2+(y-y0)^2) ;
bx1(x,y) = c1*I0*(y-y1)/((x-x1)^2+(y-y1)^2);
by1(x,y) = -c1*I0*(x-x1)/((x-x1)^2+(y-y1)^2);

bx2(x,y) = c2*I0*(y-y2)/((x-x2)^2+(y-y2)^2);
by2(x,y) = -c2*I0*(x-x2)/((x-x2)^2+(y-y2)^2);

bx3(x,y) = c3*I0*(y-y3)/((x-x3)^2+(y-y3)^2);
by3(x,y) = -c3*I0*(x-x3)/((x-x3)^2+(y-y3)^2);

bx4(x,y) = c4*I0*(y-y4)/((x-x4)^2+(y-y4)^2);
by4(x,y) = -c4*I0*(x-x4)/((x-x4)^2+(y-y4)^2);

bx(x,y) = bx0 + bx1 + bx2 + bx3 + bx4;  % Total mag fields
by(x,y) = by0 + by1 + by2 + by3 + by4;

dpsidx(x,y) = by(x,y);     % psi? 
dpsidy(x,y) = -bx(x,y);

dpdx = matlabFunction(subs(dpsidx));
dpdy = matlabFunction(subs(dpsidy));


end

function Psi = computePsi(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s)

% Computes the flux out of the variables of the problem

[X,Y]=meshgrid(xv,yv);

Psi0 = I0/2*c0*((log((X-x0).^2+(Y-y0).^2))+exp(-((X-x0).^2+(Y-y0).^2)/s^2));
Psi1 = I0/2*c1*(log((X-x1).^2+(Y-y1).^2));
Psi2 = I0/2*c2*(log((X-x2).^2+(Y-y2).^2));
Psi3 = I0/2*c3*(log((X-x3).^2+(Y-y3).^2));
Psi4 = I0/2*c4*(log((X-x4).^2+(Y-y4).^2));
Psi = Psi0 + Psi1 + Psi2 + Psi3 + Psi4;

end

function COp = optimizeKappa(kappaTarg, caux, c0, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s)

% Returns optimized value for caux (amplitude of auxiliary currents) for
% a given target kappa

% Optimizes c0 and caux

 C = [c0, caux];
 
 [COp, ~, exit] = fminsearch(@toMinimize,C);
 disp(exit)

    function cost = toMinimize(C)
        
        % Cost function to minimize
        
        c0 = C(1);
        caux = C(2);

        c1 = caux;
        c2 = caux;
        c3 = caux;
        c4 = caux;

        [~, kappa, ~, ~] = computeSpecs(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s);

        cost = abs(kappaTarg - kappa);
    
    end

end


function [delta, kappa, Rmin, Rmax] = computeSpecs(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s)

% Computes elongation out of the variables of the problem

[dpdx, dpdy] = computeDpsidr(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s);
Psi = computePsi(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s);

% Find xpoint coordinates

[iXxpt, iYxptlow] = xptCoords(xv, yv, dpdx, dpdy);
iYxptup = size(yv, 2) - iYxptlow;

Yxptlow = yv(iYxptlow);
Yxptup = yv(iYxptup);

Xxpt = xv(iXxpt);

level = Psi(iYxptlow, iXxpt);

%Obtain coordinates of points on separatrix
c = contourc(xv,yv,Psi,[level level]);
sepX = [];
sepY = [];
inb = 1;

while inb < size(c, 2)
    iend = inb + c(2, inb);
    sepX = [sepX c(1, inb+1:iend)]; 
    sepY = [sepY c(2, inb+1:iend)];
    inb = iend+1;
end



    function [Rmin, Rmax] = Rminmax(sepX,sepY, Yxptlow, Yxptup)
        i = 1;
        Rmin = inf;
        Rmax = 0;

        while i < size(sepX,2)
            if (Yxptlow <= sepY(i)) && (sepY(i) <= Yxptup)  % Checks point i is comprised between the two Xpoints
                Rmin = min(Rmin, sepX(i));
                Rmax = max(Rmax, sepX(i));
            end
            i = i+1;
        end

    end


% Rmin = min(sepX(iYxptlow:iYxptup));
% Rmax = max(sepX(iYxptlow:iYxptup));
% 
[Rmin, Rmax] = Rminmax(sepX, sepY, Yxptlow, Yxptup);

R0 = (Rmax+Rmin)/2;
a = (Rmax - Rmin)/2;

kappa = (Yxptup-Yxptlow)/(Rmax-Rmin);
delta = (R0 - Xxpt)/a;  % Assumed same X for both Xpoints


end


function XOp = optimizeDelta(deltaTarg, kappaTarg, X, x0, y0, y1, ...
    y2, x3, y3, x4, y4, I0, xv, yv,s)

% Returns optimized value for X for
% a given target kappa

XOp = fminsearch(@toMinimize, X);

    function cost = toMinimize(X)
        
        % Cost function to minimize


        x1 = X(1);
        x2 = x1;
        c1 = X(2);
        c2 = c1;
        c3 = X(3);
        c4 = c3;
        c0 = X(4);

        
        [delta, kappa, ~, ~] = computeSpecs(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv,s);

        cost = abs(deltaTarg - delta) + abs(kappaTarg - kappa);
    
    end

end


function plotMagField(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s, xv, yv)

Xaux = [x0 x1 x2 x3 x4]; % Coordinates of auxiliary currents
Yaux = [y0 y1 y2 y3 y4];

[X,Y]=meshgrid(xv,yv);

Psi = computePsi(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, xv, yv, s);

[dpdx, dpdy] = computeDpsidr(c0, c1, c2, c3, c4, x0, y0, x1, y1, ...
    x2, y2, x3, y3, x4, y4, I0, s);

[iXxpt, iYxptlow] = xptCoords(xv, yv, dpdx, dpdy);
iYxptup = size(yv, 2) - iYxptlow;

Yxptlow = yv(iYxptlow);
Yxptup = yv(iYxptup);

Xxpt = xv(iXxpt);

figure;
contour(X,Y,Psi,Psi(end/2,end/2):Psi(end/2,end/2)/200:Psi(end/2,end-5),'k')
hold on

for aux = 1:size(Xaux,2)   % Plot auxiliary currents                                        
    plot(Xaux(aux), Yaux(aux), 'o')
end

%Obtain coordinates of points on separatrix
c = contour(X,Y,Psi,'r','LevelList',Psi(iYxptlow,iXxpt));

xlabel('$R/\rho_{s0}$','Interpreter','latex')
ylabel('$Z/\rho_{s0}$','Interpreter','latex')
title('$\Psi$','Interpreter','latex')
axis equal

end

