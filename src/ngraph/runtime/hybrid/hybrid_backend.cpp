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

#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/interpreter/int_placement.hpp"
#include "ngraph/util.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::hybrid::HYBRIDBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

shared_ptr<runtime::Tensor> runtime::hybrid::HYBRIDBackend::create_tensor(const element::Type& type,
                                                                          const Shape& shape)
{
    return make_shared<runtime::HostTensor>(type, shape, "external");
}

shared_ptr<runtime::Tensor> runtime::hybrid::HYBRIDBackend::create_tensor(const element::Type& type,
                                                                          const Shape& shape,
                                                                          void* memory_pointer)
{
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer, "external");
}

bool runtime::hybrid::HYBRIDBackend::compile(shared_ptr<Function> function)
{
    NGRAPH_INFO << "hybrid compile -Begin ";
    if (m_function_map.find(function) == m_function_map.end())
    {
        // Clone function
        FunctionInstance instance;
        instance.m_function = clone_function(*function);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::AssignPlacement>(
            runtime::interpreter::default_placement_policy);
        pass_manager.run_passes(instance.m_function);

        NGRAPH_INFO << "hybrid compile -begin split  ";
        // Split function to sub_functions
        tie(instance.m_sub_functions, instance.m_map_parameter_to_result) =
            split_function_by_placement(instance.m_function);
        NGRAPH_INFO << "hybrid compile -End split  ";

        m_function_map.insert({function, instance});
        NGRAPH_INFO << "hybrid compile -map incertion successful";
    }
    NGRAPH_INFO << "hybrid compile -End ";
    return true;
}

bool runtime::hybrid::HYBRIDBackend::call(shared_ptr<Function> function,
                                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    NGRAPH_INFO << "hybrid call -Begin ";

    validate_call(function, outputs, inputs);
    compile(function);

    NGRAPH_INFO << "hybrid call -End ";
    return true;
}
