#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <VoxieClient/Array.hpp>
#include <VoxieClient/ClaimedOperation.hpp>
#include <VoxieClient/DBusTypeList.hpp>

#include <array>
#include <cmath>
#include <iostream>

#include <QObject>

/*
* CNN-AI-CT class which provides model specific inference from a given input and
* output voxie volume.
*
* based on
* 
* BEAM HARDENING ARTIFACT REDUCTION IN X-RAY CT RECONSTRUCTION OF
* 3D PRINTED METAL PARTS LEVERAGING DEEP LEARNING AND CAD MODELS
* Ziabari et al., Proceedings of the ASME 2020 International Mechanical Engineering Congress and Exposition
* https://www.osti.gov/biblio/1769244
*
* Requires Libtorch C++ in build to build and infere pytorch models
*/
class CNNAiCt {
 public:
  CNNAiCt(torch::jit::script::Module module);
  void infere(vx::Array3<const float>& inputVolume,
              vx::Array3<float>& outputVolume, int batchSize,
              vx::ClaimedOperation<
                  de::uni_stuttgart::Voxie::ExternalOperationRunFilter>& prog);

 private:
  torch::jit::script::Module module;
};
