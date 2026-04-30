%clear; clc; close all;

% =========================
% INPUT PARAMETERS
% =========================

% Nuclear device
Y = 1;                          % Yield [megatons]
f_rad = 0.35;                   % Fraction of energy in radiation
d = 5000;                       % Standoff distance [m]

% Geometry

% Example Omega value; commented out to try solving for Omega using dimensions
%Omega = 0.0012;                   % Solid angle [sr] (example)

% Asteroid properties
density = 2206.84961424523;     % kg/m^3
porosity = 0.366370558096897;   % fraction (0–1)
taxonomy = 'S';                 % 'C', 'S', or 'M'

% Shape (ellipsoid)
a = 114.3077147;                % m
b = 120.614479;                 % m
c = 228.0234134;                % m


% =========================
% ATEROID SIZE
% =========================

R_ast = ((a*b*c)^(1/3))/2;      % radius


% =========================
% COMPUTE SOLID ANGLE
% =========================

theta = acos(R_ast / (R_ast+d));
Omega = 2*pi*(1-cos(theta));


% =========================
% MATERIAL MODEL
% =========================

switch taxonomy
    case 'C'  % Carbonaceous
        Q_star = 5e6;
        eta_kin = 0.2;
    case 'S'  % Stony
        Q_star = 8.6;
        eta_kin = 0.3;
    case 'M'  % Metallic
        Q_star = 1.2e7;
        eta_kin = 0.4;
    otherwise
        error('Unknown taxonomy');
end

eta_abs = 0.5 * (1 - porosity);


% =========================
% ENERGY
% =========================

E_total = Y * 4.184e15;         % [J]
E_rad = f_rad * E_total;

E_incident = E_rad * (Omega / (4*pi));


% =========================
% EJECTA VELOCITY
% =========================

v_ej = sqrt(2 * eta_kin * Q_star);


% =========================
% COUPLING COEFFICIENT
% =========================

Cm = (eta_abs * v_ej) / Q_star;


% =========================
% ASTEROID MASS
% =========================

rho_eff = density * (1 - porosity);
Volume = (4/3)*pi*a*b*c;
M_ast = rho_eff * Volume;


% =========================
% DELTA V
% =========================

deltaV = (Cm * E_incident) / M_ast;


% =========================
% OUTPUT
% =========================

fprintf('Standoff distance: %.1f m\n', d);
fprintf('Solid angle: %.3f sr\n', Omega);
fprintf('Ejecta velocity: %.1f m/s\n', v_ej);
fprintf('Cm: %.3e N*s/J\n', Cm);
fprintf('Delta-V: %.6f m/s (%.3f mm/s)\n', deltaV, deltaV*1e3);
