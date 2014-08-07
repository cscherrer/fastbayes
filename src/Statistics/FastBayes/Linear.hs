{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE BangPatterns #-}

{-|
Module      : Statistics.FastBayes.Linear
Description : Bayesian linear regression via maximum marginal likelihood.
Copyright   : (c) Melinae, 2014
                  Chad Scherrer, 2014
License     : MIT
Maintainer  : chad.scherrer@gmail.com
Stability   : experimental
Portability : POSIX

This module gives an implementation of Bayesian linear regression, with the scale of the prior chosen by marginal likelihood.

The inputs for a Bayesian linear model are identical to those of a classical linear model, except that in addition to a design matrix and response, we must also specify a prior distribution on the weights and the noise. This leaves us with an open question of how these should be specified.

In his book /Pattern Recognition and Machine Learning/, Christopher Bishop provides details for an approach that simplifies the situation significantly, and allows for much faster inference. The structure of the linear model allows us to integrate the posterior over the weights, resulting in the /marginal likelihood/, expressed as a function of the prior precision and noise precision. This, in turn, can be easily optimized.
-}

module Statistics.FastBayes.Linear 
  ( Fit
  , marginalLikelihood
  , design
  , response
  , priorPrecision        
  , noisePrecision        
  , numEffectiveParameters
  , logEvidence           
  , mapWeights            
  , hessian
  )
  where

import qualified Data.Vector.Storable as V
import Numeric.LinearAlgebra


data Fit = Fit
  { design                 :: Matrix Double -- ^The design matrix used for the fit. 
  , response               :: Vector Double -- ^The response vector used for the fit
  , priorPrecision         :: Double        -- ^The precision (inverse variance) of the prior distribution, determined by maximizing the marginal likelihood
  , noisePrecision         :: Double        -- ^The precision (inverse variance) of the noise
  , numEffectiveParameters :: Double        -- ^The number of effective parameters in the model
  , logEvidence            :: Double        -- ^The log of the evidence, which is useful for model comparison (different features, same response)
  , mapWeights             :: Vector Double -- ^The MAP (maximum a posteriori) values for the paramter weights
  , hessian                :: Matrix Double -- ^The Hessian (matrix of second derivatives) for the posterior distribution
  }
  deriving Show


marginalLikelihood :: 
  ([(Double, Double)] → (Double, Double))   -- How to take the limit of the (α,β) sequence
  → Matrix Double                           -- design matrix (features in columns)
  → Vector Double                           -- response vector
  → Fit
marginalLikelihood lim x y = Fit x y α β γ logEv m h
  where
  n = rows x
  p = cols x
  α0 = 1.0
  β0 = 1.0

  -- A vector of the eigenvalues of xtx
  (_,sqrtEigs,_) = compactSVD x
  eigs = V.map square sqrtEigs

  xtx = trans x <> x
  xty = trans x <> y
  getHessian a b = diag (V.replicate p a) + scale b xtx

  m = scale β $ h <\> xty
  
  go :: Double → Double → [(Double, Double)]
  go a0 b0 = (a0, b0) : go a b
    where
    h0 = getHessian a0 b0
    m0 = scale b0 $ h0 <\> xty 
    c = V.sum $ V.map (\x' → x' / (a0 + x')) eigs
    a = c / (m0 <.> m0)
    b = recip $ (normSq $ y - x <> m0) / (fromIntegral n - c)
  
  γ = V.sum $ V.map (\x' → x' / (α + x')) eigs

  h = getHessian α β 
  logEv = 0.5 * 
    ( fromIntegral p * log α 
    + fromIntegral n * log β 
    - (β * normSq (y - x <> m) + α * (m <.> m))
    - logDetH 
    - fromIntegral n * log (2*pi)
    )
    where
    (_,(logDetH, _)) = invlndet h

  (α, β) = lim $ go α0 β0

square :: Double → Double
square x = x * x

normSq :: Vector Double → Double
normSq x = x <.> x