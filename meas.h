#pragma once
#include "itensor/mps/dmrg.h"

namespace itensor {

inline IQTensor 
multSite(IQTensor a, IQTensor b) 
    {
    IQTensor res = a * b;
    res.mapprime(1,0,Site);
    return res;
    }

class Meas
    {
public:
    Hubbard & sp;
    IQMPS psi, psip;
    std::vector<IQTensor> Left, Right;
    int N;
    bool prepped;
    int corrdist;
    Matrix  cdagc, ccdag;
    Matrix  cdagcdn, ccdagdn;
    Matrix  nn;
    Meas(Hubbard& sp_,const IQMPS& psip_, const IQMPS& psi_,int cd=100)
		: sp(sp_), psi(psi_), psip(psip_),  prepped(false), corrdist(cd)
        {
        psip.mapprime(0,1,Link);
        N = psi.N();
        Left.resize(N+1);
        Right.resize(N+1);
        Right[N] = dag(psip.A(N)) * psi.A(N);
        Left[1]  = dag(psip.A(1)) * psi.A(1);
        for(int i = N-1; i >= 2; i--)
            Right[i] = Right[i+1] * dag(psip.A(i)) * psi.A(i);
        for(int i = 2; i < N; i++)
            Left[i] =  Left[i-1] * dag(psip.A(i)) * psi.A(i);
        }

    Real Ntot(int i) const
        {
        IQTensor R = dag(psip.A(i));
        if(i < N) R = R * Right[i+1];
        IQTensor R2 = multSite(sp.op("Ntot",i),psi.A(i));
        R2.mapprime(1,0,Site);
        if(i > 1) R2 = R2 * Left[i-1];
        R *= R2;
        auto c = R.cplx();
        return real(c);
        }
    Real Nup(int i) const
        {
        IQTensor R = dag(psip.A(i));
        if(i < N) R = R * Right[i+1];
        IQTensor R2 = multSite(sp.op("Nup",i),psi.A(i));
        R2.mapprime(1,0,Site);
        if(i > 1) R2 = R2 * Left[i-1];
        R *= R2;
        auto c = R.cplx();
        return real(c);
        }
    Real Ndn(int i) const
        {
        IQTensor R = dag(psip.A(i));
        if(i < N) R = R * Right[i+1];
        IQTensor R2 = multSite(sp.op("Ndn",i),psi.A(i));
        R2.mapprime(1,0,Site);
        if(i > 1) R2 = R2 * Left[i-1];
        R *= R2;
        auto c = R.cplx();
        return real(c);
        }
    Real Nupdn(int i) const
        {
        IQTensor R = dag(psip.A(i));
        if(i < N) R = R * Right[i+1];
        IQTensor R2 = multSite(sp.op("Nupdn",i),psi.A(i));
        R2.mapprime(1,0,Site);
        if(i > 1) R2 = R2 * Left[i-1];
        R *= R2;
        auto c = R.cplx();
        return real(c);
        }
    void prep2site()
        {
        if(prepped) return;
        prepped = true;
        resize(cdagc,N,N);
        nn = cdagcdn = ccdagdn = ccdag = cdagc;
        for(int i = 1; i < N; i++)
            {
            int jdo = std::min(N,i+corrdist);
            IQTensor R2 = multSite(sp.op("Ntot",i),psi.A(i));
            if(i > 1) R2 = R2 * Left[i-1];
            R2 = R2 * dag(psip.A(i));
            for(int j = i+1; j <= jdo; j++)
                {
                IQTensor R = multSite(sp.op("Ntot",j),psi.A(j));
                if(j < N) R = R * Right[j+1];
                R = R * dag(psip.A(j));
                nn(i-1,j-1) = (R*R2).real();
                R2 =  R2 * dag(psip.A(j)) * psi.A(j); 
                }

            R2 = multSite(sp.op("Cdagup",i),multSite(sp.op("F",i),psi.A(i)));
            if(i > 1) R2 = R2 * Left[i-1];
            R2 = R2 * dag(psip.A(i));
            for(int j = i+1; j <= jdo; j++)
                {
                IQTensor R = multSite(sp.op("Cup",j),psi.A(j));
                if(j < N) R = R * Right[j+1];
                R = R * dag(psip.A(j));
                cdagc(i-1,j-1) = (R*R2).real();
                R2 =  R2 * dag(psip.A(j)) * multSite(sp.op("F",j),psi.A(j)); 
                }

            R2 = multSite(sp.op("Cup",i),multSite(sp.op("F",i),psi.A(i)));
            if(i > 1) R2 = R2 * Left[i-1];
            R2 = R2 * dag(psip.A(i));
            for(int j = i+1; j <= jdo; j++)
                {
                IQTensor R = multSite(sp.op("Cdagup",j),psi.A(j));
                if(j < N) R = R * Right[j+1];
                R = R * dag(psip.A(j));
                ccdag(i-1,j-1) = (R*R2).real();
                R2 =  R2 * dag(psip.A(j)) * multSite(sp.op("F",j),psi.A(j)); 
                }

            R2 = multSite(sp.op("Cdagdn",i),multSite(sp.op("F",i),psi.A(i)));
            if(i > 1) R2 = R2 * Left[i-1];
            R2 = R2 * dag(psip.A(i));
            for(int j = i+1; j <= jdo; j++)
                {
                IQTensor R = multSite(sp.op("Cdn",j),psi.A(j));
                if(j < N) R = R * Right[j+1];
                R = R * dag(psip.A(j));
                cdagcdn(i-1,j-1) = (R*R2).real();
                R2 =  R2 * dag(psip.A(j)) * multSite(sp.op("F",j),psi.A(j)); 
                }

            R2 = multSite(sp.op("Cdn",i),multSite(sp.op("F",i),psi.A(i)));
            if(i > 1) R2 = R2 * Left[i-1];
            R2 = R2 * dag(psip.A(i));
            for(int j = i+1; j <= jdo; j++)
                {
                IQTensor R = multSite(sp.op("Cdagdn",j),psi.A(j));
                if(j < N) R = R * Right[j+1];
                R = R * dag(psip.A(j));
                ccdagdn(i-1,j-1) = (R*R2).real();
                R2 =  R2 * dag(psip.A(j)) * multSite(sp.op("F",j),psi.A(j)); 
                }
            }

        }
    Real CdagC(int i, int j)
        {
        if(i == j) return Nup(i);
        if(i > j) return -CCdag(j,i);
        if(!prepped)
            prep2site();
        return cdagc(i-1,j-1);
        }
    Real CCdag(int i, int j)
        {
        if(i == j) return 1.0 - Nup(i);
        if(i > j) return -CdagC(j,i);
        if(!prepped)
            prep2site();
        return ccdag(i-1,j-1);
        }
    Real CdagCdn(int i, int j)
        {
        if(i == j) return Ndn(i);
        if(i > j) return -CCdagdn(j,i);
        if(!prepped)
            prep2site();
        return cdagcdn(i-1,j-1);
        }
    Real CCdagdn(int i, int j)
        {
        if(i == j) return 1.0 - Ndn(i);
        if(i > j) return -CdagCdn(j,i);
        if(!prepped)
            prep2site();
        return ccdagdn(i-1,j-1);
        }
    Real NN(int i, int j)
        {
        if(i == j) return Nupdn(i);
        if(!prepped)
            prep2site();
        return nn(i-1,j-1);
        }
    };

} //namespace itensor
