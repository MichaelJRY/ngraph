//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// Wrapper for ONNXIFI output tensor
        class OutputTensor final : public Tensor
        {
        public:
            using Tensor::Tensor;
            std::shared_ptr<runtime::Tensor> to_ng(runtime::Backend& backend) const final;
            void from_ng(const runtime::Tensor& tensor);
        };

    } // namespace onnxifi

} // namespace ngraph
