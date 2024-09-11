#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double Tree(double S, double K, double r, double q, double v, double t, int steps, int optype)
{
    double dt = t / (double) steps;

    int row = 4*steps + 2, column = steps + 1;
    
    double tree[row][column];

    int center = row / 2 - 1;

    double up = exp(v*sqrt(2.0*dt));
    double dn = 1.0/up;
    double m = 1.0;

    double A = exp((r - q)*dt/2.0);
    double B = exp(-v*sqrt(dt/2.0));
    double C = exp(v*sqrt(dt/2.0));

    double pu = pow((A - B)/(C - B), 2);
    double pd = pow((C - A)/(C - B), 2);
    double pm = 1.0 - (pu + pd);

    for(int j = 0; j < column; ++j){
        tree[center][j] = S;
        for(int i = 1; i < column - j; ++i){
            tree[center - 2*i][i + j] = tree[center - 2*(i-1)][i - 1 + j]*up;
            tree[center][i + j] = tree[center][i - 1 + j]*m;
            tree[center + 2*i][i + j] = tree[center + 2*(i-1)][i - 1 + j]*dn;
        }
    }

    for(int i = 0; i < row; ++i){
        if(i % 2 != 0){
            if(optype == 0){
                tree[i][column - 1] = fmax(tree[i-1][column-1] - K, 0.0);
            } else {
                tree[i][column - 1] = fmax(K - tree[i-1][column-1], 0.0);
            }
        }
    }

    int inc = 2;
    for(int j = 2; j < column + 1; ++j){
        for(int i = inc; i < row - inc; ++i){
            if(i % 2 != 0){
                A = tree[i-2][column-j+1];
                B = tree[i][column-j+1];
                C = tree[i+2][column-j+1];
                double cash = pu*A + pm*B + pd*C;
                cash = exp(-r*dt)*cash;
                if(optype == 0){
                    tree[i][column-j] = fmax(tree[i-1][column-j] - K, cash);
                } else {
                    tree[i][column-j] = fmax(K - tree[i-1][column-j], cash);
                }
            }
        }
        inc += 2;
    }

    return tree[center + 1][0];
}

double IV(double opPrice, double S, double K, double r, double q, double t, int steps, int optype)
{
    double v0 = 0.5, v1 = 0.99;
    double dV = 0.01;
    while(dV != 0.02){
        double vega = (Tree(S, K, r, q, v0+dV, t, steps, optype) - Tree(S, K, r, q, v0-dV, t, steps, optype))/(2.0*dV);
        double price = Tree(S, K ,r, q, v0, t, steps, optype);
        v1 = v0 - 0.1*(price - opPrice)/vega;
        if(isnan(v1) || isinf(v1)){
            //printf("P: %f\n", v1);
            return 0;
        } else {
            //printf("Q: %f\n", v1);
        }
        if(fabs(v1 - v0) < 0.0001){
            break;
        }
        v0 = v1;
    }
    v1 = fmax(v1, 0.0);
    return v1;
}

double Delta(double S, double K, double r, double q, double v, double t, int steps, int optype)
{
    double dS = 0.01*S;
    double delta = (Tree(S+dS, K, r, q, v, t, steps, optype) - Tree(S-dS, K, r, q, v, t, steps, optype))/(2.0*dS);
    return delta;
}

double Gamma(double S, double K, double r, double q, double v, double t, int steps, int optype)
{
    double dS = 0.01*S;
    double gamma = (Tree(S+dS, K, r, q, v, t, steps, optype) - 2.0*Tree(S, K, r, q, v, t, steps, optype) + Tree(S-dS, K, r, q, v, t, steps, optype))/pow(dS, 2);
    return gamma;
}

double Theta(double S, double K, double r, double q, double v, double t, int steps, int optype)
{
    double dT = 1.0/365.0;
    double theta = -(Tree(S, K, r, q, v, t+dT, steps, optype) - Tree(S, K, r, q, v, t, steps, optype))/dT;
    return theta/252.0;
}

double Vega(double S, double K, double r, double q, double v, double t, int steps, int optype)
{
    double dV = 0.01;
    double vega = (Tree(S, K, r, q, v+dV, t, steps, optype) - Tree(S, K, r, q, v-dV, t, steps, optype))/(2.0*dV);
    return fmin(fmax(vega/100.0, 0), 1.0);
}

double Rho(double S, double K, double r, double q, double v, double t, int steps, int optype)
{
    double dR = 0.01;
    double rho = (Tree(S, K, r+dR, q, v, t, steps, optype) - Tree(S, K, r-dR, q, v, t, steps, optype))/(2.0*dR);
    return rho/100.0;
}

double Vanna(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dS = 0.01*S;
    double p0 = Vega(S-dS, K, r, q, v, t, steps, optype);
    double p1 = Vega(S+dS, K, r, q, v, t, steps, optype);
    return (p1 - p0)/(2.0*dS);
}

double DeltaDecay(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dT = 1.0/365.0;
    double p0 = Delta(S, K, r, q, v, t, steps, optype);
    double p1 = Delta(S, K, r, q, v, t+dT, steps, optype);
    return -(p1 - p0)/dT;
}

double Volga(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dV = 0.01;
    double p0 = Vega(S, K , r, q, v-dV, t, steps, optype);
    double p1 = Vega(S, K , r, q, v+dV, t, steps, optype);
    return (p1 - p0)/(2.0*dV);
}

double Veta(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dT = 1.0/365.0;
    double p0 = Vega(S, K, r, q, v, t, steps, optype);
    double p1 = Vega(S, K, r, q, v, t+dT, steps, optype);
    return (p1 - p0)/dT;
}

double GammaDecay(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dT = 1.0/365.0;
    double p0 = Gamma(S, K, r, q, v, t, steps, optype);
    double p1 = Gamma(S, K, r, q, v, t+dT, steps, optype);
    return -(p1 - p0)/dT;
}

double Zomma(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dV = 0.01;
    double p0 = Gamma(S, K, r, q, v-dV, t, steps, optype);
    double p1 = Gamma(S, K, r, q, v+dV, t, steps, optype);
    return (p1 - p0)/(2.0*dV);
}

double Speed(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dS = 0.01*S;
    double p0 = Gamma(S-dS, K, r, q, v, t, steps, optype);
    double p1 = Gamma(S+dS, K, r, q, v, t, steps, optype);
    return (p1 - p0)/(2.0*dS);
}

double Ultima(double S, double K, double r, double q, double v, double t, int steps, int optype){
    double dV = 0.01;
    double p0 = Volga(S, K, r, q, v-dV, t, steps, optype);
    double p1 = Volga(S, K, r, q, v+dV, t, steps, optype);
    return (p1 - p0)/(2.0*dV);
}
