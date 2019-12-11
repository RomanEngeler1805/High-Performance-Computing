/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

template
<
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Conv2DLayer: public Layer
{
  Params* allocate_params() const override {
    //number of kernel parameters:
    // 2d kernel size * number of inp channels * number of out channels
    const int nParams = KnY * KnX * InC * KnC;
    const int nBiases = KnC;
    return new Params(nParams, nBiases);
  }

  Conv2DLayer(const int _ID) : Layer(OpX * OpY * KnC, _ID) {
    static_assert(InX>0 && InY>0 && InC>0, "Invalid input");
    static_assert(KnX>0 && KnY>0 && KnC>0, "Invalid kernel");
    static_assert(OpX>0 && OpY>0, "Invalid outpus");
    print();
  }

  void print() {
    printf("(%d) Conv: In:[%d %d %d %d %d] F:[%d %d %d %d] Out:[%d %d %d]\n",
      ID, OpY,OpX,KnY,KnX,InC, KnY,KnX,InC,KnC, OpX,OpY,KnC);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    assert(act[ID]->layersSize   == OpY * OpX *                   KnC); // nOutput
    assert(act[ID-1]->layersSize == OpY * OpX * KnY * KnX * InC      ); // nInput, from before performed im2mat
    assert(param[ID]->nWeights   ==             KnY * KnX * InC * KnC);
    assert(param[ID]->nBiases    ==                               KnC);

    const int batchSize = act[ID]->batchSize;
    const Real* const INP = act[ID-1]->output; // (OpY * OpX) * (KnY * KnX * InC)
          Real* const OUT = act[ID]->output;  // (OpY * OpX) * (KnC)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    const Real*const WEIGHT = param[ID]->weights; //size is (KnY * KnX) * (InC * KnC)
    const Real*const BIAS   = param[ID]->biases; //size is KnC

    const int nOutput = act[ID]->layersSize;
    const int nInput = act[ID]->layersSize;

    // initializing with bias
#pragma omp parallel for
    for(int b= 0; b< batchSize; b++)
      for(int opy= 0; opy< OpY; opy++)
        for(int opx= 0; opx< OpX; opx++)
          for(int knc= 0; knc< KnC; knc++)
            OUT[b*OpY*OpX*KnC + opy*OpX* KnC+ KnC*opx+ knc] = BIAS[knc];

    // convolution
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize* OpY * OpX, KnC, KnY * KnX * InC, 1.0, INP, KnY * KnX * InC, WEIGHT, KnC, 1.0, OUT, KnC);
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad) const override
  {
    const int batchSize = act[ID]->batchSize;
    const Real* const dEdO = act[ID]->dError_dOutput;

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    const Real* const INP = act[ID-1]->output; // (OpY * OpX) * (KnY * KnX * InC)
    const Real* const WEIGHTS = param[ID]->weights; // (KnY * KnX) * (InC * KnC)

    const int nInputs = act[ID-1]->layersSize;
    const int nOutputs = act[ID]->layersSize;

    // Bias gradient
    Real* const grad_B = grad[ID]->biases; // (KnC)

#pragma omp parallel for
    for(int knc= 0; knc< KnC; knc++) grad_B[knc] = 0;

#pragma omp parallel for
    for(int knc= 0; knc< KnC; knc++)
      for(int i= 0; i< batchSize* OpY* OpX; i++)
        grad_B[knc] += dEdO[i* KnC+ knc];

    // Weight gradient
    Real* const grad_W = grad[ID]->weights; // size (KnY * KnX * InC) * (KnC)
    //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    //            KnY * KnX * InC, OpY * OpX, batchSize, 1.0, INP, KnY * KnX * InC, dEdO, OpY * OpX, 0.0, grad_W, OpY * OpX);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                KnY * KnX * InC, KnC, batchSize* OpY * OpX, 1.0, INP, KnY * KnX * InC, dEdO, KnC, 0.0, grad_W, KnC);

    // Previous layer gradient
    Real* const errinp = act[ID-1]->dError_dOutput; // batchSize * nInputs

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batchSize* OpY * OpX, KnY * KnX * InC, KnC, 1.0, dEdO, KnC, WEIGHTS, KnC, 0.0, errinp, KnY * KnX * InC);

    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //            batchSize, nInputs, nOutputs, 1.0, dEdO, nOutputs, WEIGHTS, nOutputs, 0.0, errinp, nInputs);

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  }

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override
  {
    // get pointers to layer's weights and bias
    Real *const W = param[ID]->weights, *const B = param[ID]->biases;
    // initialize weights with Xavier initialization
    const int nAdded = KnX * KnY * InC, nW = param[ID]->nWeights;
    const Real scale = std::sqrt(6.0 / (nAdded + KnC));
    std::uniform_real_distribution < Real > dis(-scale, scale);
    std::generate(W, W + nW, [&]() {return dis( gen );});
    std::fill(B, B + KnC, 0);
  }
};
