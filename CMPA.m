close all;
clear;
clc;

Is = 0.01e-12; %pA
Ib = 0.1e-12; %pA
Vb = 1.3; %V
Gp = 0.1; %Ohm^-1

Diode_model = @(V) (Is.*(exp((1.2/0.025).*V) - 1)) + (Gp.*V) - (Ib.*(exp((-1.2/0.025).*(V + Vb)) - 1));

% --------- Generate some data  --------- %
voltage = linspace(-1.95, 0.7, 200);
I_noise = zeros(1, length(voltage));

I = Diode_model(voltage);

for i = 1:length(I_noise)
   
    I_noise(i) = I(i)*0.8 + (0.4)*I(i)*rand(1);
    
end

I_with_noise = I + I_noise;

figure;

subplot(3, 1, 1);
plot(voltage, I, 'r');
hold on;
plot(voltage, I_noise, 'b.');
hold off;
grid on;
title('Current and Noise Separately (Linear Plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Random Noise');

subplot(3, 1, 2);
semilogy(voltage, abs(I), 'r');
hold on;
semilogy(voltage, abs(I_noise), 'b.');
grid on;
title('Current and Noise Separately (Semilog Plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Random Noise');

subplot(3, 1, 3);
plot(voltage, (I_with_noise), 'r');
grid on;
title('Current and Noise Together (Linear Plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');

% --------- Polynomial Fitting  --------- %

polyfit4 = polyfit(voltage, I_with_noise, 4);
polyfit8 = polyfit(voltage, I_with_noise, 8);

figure;

subplot(2, 1, 1);
plot(voltage, I_with_noise, 'r');
hold on;
plot(voltage, polyval(polyfit4, voltage), 'b');
hold on;
plot(voltage, polyval(polyfit8, voltage), 'g');
hold off;
grid on;
title('Current with Polynomial Fits (Linear)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', '4th Order Polynomial Fit', '8th Order Polynomial Fit');

subplot(2, 1, 2);
semilogy(voltage, abs(I_with_noise), 'r');
hold on;
semilogy(voltage, abs(polyval(polyfit4, voltage)), 'b');
hold on;
semilogy(voltage, abs(polyval(polyfit8, voltage)), 'g');
hold off;
grid on;
title('Current with Polynomial Fits (Semilog)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', '4th Order Polynomial Fit', '8th Order Polynomial Fit');

% --------- Nonlinear Fitting  --------- %

modelFunc1 = fittype('A.*(exp(1.2*x/0.025)-1) + B.*x - C*(exp(1.2*(-(x+D))/0.025)-1)');
fitParams1 = fit(transpose(voltage), transpose(I_with_noise), modelFunc1);
currentFit1 = fitParams1(voltage);

figure;

subplot(3, 2, 1);
plot(voltage, I_with_noise, 'r');
hold on;
plot(voltage, currentFit1, 'b');
hold off;
grid on;
title('Current with Nonlinear Fit (B and D explicitely set - Linear plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Nonlinear Fit');

subplot(3, 2, 2);
semilogy(voltage, abs(I_with_noise), 'r');
hold on;
semilogy(voltage, abs(currentFit1), 'b');
hold off;
grid on;
title('Current with Nonlinear Fit (B and D explicitely set - Semilog plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Nonlinear Fit');

modelFunc2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
fitParams2 = fit(transpose(voltage), transpose(I_with_noise), modelFunc2);
currentFit2 = fitParams2(voltage);

subplot(3, 2, 3);
plot(voltage, I_with_noise, 'r');
hold on;
plot(voltage, currentFit2, 'b');
hold off;
grid on;
title('Current with Nonlinear Fit (D explicitely set - Linear plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Nonlinear Fit');

subplot(3, 2, 4);
semilogy(voltage, abs(I_with_noise), 'r');
hold on;
semilogy(voltage, abs(currentFit2), 'b');
hold off;
grid on;
title('Current with Nonlinear Fit (D explicitely set - Semilog plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Nonlinear Fit');

modelFunc3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
fitParams3 = fit(transpose(voltage), transpose(I_with_noise), modelFunc3);
currentFit3 = fitParams3(voltage);

subplot(3, 2, 5);
plot(voltage, I_with_noise, 'r');
hold on;
plot(voltage, currentFit3, 'b');
hold off;
grid on;
title('Current with Nonlinear Fit (Linear plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Nonlinear Fit');

subplot(3, 2, 6);
semilogy(voltage, abs(I_with_noise), 'r');
hold on;
semilogy(voltage, abs(currentFit3), 'b');
hold off;
grid on;
title('Current with Nonlinear Fit (Semilog plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Nonlinear Fit');

% --------- Neural Net Fitting --------- %

inputs = voltage;
targets = I_with_noise;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

figure;

subplot(2, 1, 1);
plot(voltage, I_with_noise, 'r');
hold on;
plot(voltage, Inn, 'b');
grid on;
title('Current with Neural Net Fit (Linear plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Neural Net Fit');

subplot(2, 1, 2);
semilogy(voltage, abs(I_with_noise), 'r');
hold on;
semilogy(voltage, abs(Inn), 'b');
grid on;
title('Current with Neural Net Fit (Semilog plot)');
xlabel('Voltage (V)');
ylabel('Current (A)');
legend('Diode Current', 'Neural Net Fit');