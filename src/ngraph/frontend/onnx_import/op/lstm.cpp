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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/type/element_type.hpp"

#include "exceptions.hpp"
#include "lstm.hpp"
#include "utils/broadcasting.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

                NgraphNodePtr squeeze_first_dim(const NgraphNodePtr& node)
                {
                    auto data_shape{node->get_shape()};
                    Shape output_shape(std::next(std::begin(data_shape)), std::end(data_shape));
                    return std::make_shared<ngraph::op::Reshape>(
                        node,
                        reshape::get_default_axis_vector(data_shape.size()),
                        output_shape);
                }

                inline std::shared_ptr<ngraph::op::Slice>
                make_ng_slice(const std::shared_ptr<ngraph::Node>& node,
                              std::vector<std::size_t> axes,
                              std::vector<std::size_t> starts,
                              std::vector<std::size_t> ends)
                {
                    std::vector<std::size_t> upper_bounds{node->get_shape()};
                    std::vector<std::size_t> lower_bounds(upper_bounds.size());
                    for (std::size_t index{0}; index < axes.size(); ++index)
                    {
                        std::size_t axis{axes.at(index)};
                        lower_bounds.at(axis) = starts.at(index);
                        upper_bounds.at(axis) = ends.at(index);
                    }
                    return std::make_shared<ngraph::op::Slice>(node, lower_bounds, upper_bounds);
                }

                NodeVector split(const NgraphNodePtr& node,
                                 std::size_t split_parts,
                                 int axis = 0,
                                 bool flatten = false)
                {
                    // TODO: refactor! Mostly copy-paste from split.cpp::split

                    std::size_t axis_to_split{static_cast<std::size_t>(axis)};
                    if (axis < 0)
                    {
                        axis_to_split = node->get_shape().size() + axis;
                    }
                    std::size_t length_axis_to_split{node->get_shape().at(axis_to_split)};
                    std::vector<std::size_t> length_parts(split_parts, length_axis_to_split / split_parts);

                    std::size_t start_index{0};
                    NodeVector outputs;
                    for (const auto& length_part : length_parts)
                    {
                        std::size_t end_index{start_index + length_part};
                        std::shared_ptr<ngraph::Node> sliced_node = make_ng_slice(node,
                                                                                  {axis_to_split},
                                                                                  {start_index},
                                                                                  {end_index});
                        start_index = end_index;
                        if (flatten)
                        {
                            auto sliced_shape{sliced_node->get_shape()};
                            Shape output_shape{std::next(std::begin(sliced_shape)), std::end(sliced_shape)};
                            sliced_node = std::make_shared<ngraph::op::Reshape>(
                                sliced_node,
                                reshape::get_default_axis_vector(sliced_shape.size()),
                                output_shape);
                        }
                        outputs.push_back(sliced_node);
                    }
                    return outputs;
                }

                NgraphNodePtr add(const NgraphNodePtr& lhs, const NgraphNodePtr& rhs)
                {
                    auto args = numpy_style_broadcast_for_binary_operation(lhs, rhs);
                    return {std::make_shared<ngraph::op::Add>(args.at(0), args.at(1))};
                }

                NgraphNodePtr mul(const NgraphNodePtr& lhs, const NgraphNodePtr& rhs)
                {
                    auto args = numpy_style_broadcast_for_binary_operation(lhs, rhs);
                    return {std::make_shared<ngraph::op::Multiply>(args.at(0), args.at(1))};
                }

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                NgraphNodePtr Sigmoid(const NgraphNodePtr& arg)
                {
                    return std::make_shared<ngraph::op::Sigmoid>(arg);
                }

                NgraphNodePtr Tanh(const NgraphNodePtr& arg)
                {
                    return std::make_shared<ngraph::op::Tanh>(arg);
                }

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                struct LSTMNgInputMap
                {
                    using iterator = std::map<std::string, NgraphNodePtr>::iterator;

                    explicit LSTMNgInputMap(const Node& node)
                    {
                        const auto& ng_inputs = node.get_ng_inputs();
                        const std::size_t gates_count{4};
                        const std::size_t peepholes_count{3};

                        // Mandatory inputs
                        // Packed input sequences. Shape: [seq_length, batch_size, input_size]
                        m_map["X"] = ng_inputs.at(0);
                        // Weight tensor for the gates. Shape: [num_directions, 4*hidden_size, input_size]
                        m_map["W"] = ng_inputs.at(1);
                        // The recurrence weight tensor. Shape: [num_directions, 4*hidden_size, hidden_size]
                        m_map["R"] = ng_inputs.at(2);

                        const std::size_t hidden_size = m_map["R"]->get_shape().back();
                        const std::size_t batch_size = m_map["X"]->get_shape().at(1);
                        const std::size_t n_dirs = m_map["W"]->get_shape().front();

                        // Optional inputs
                        // The bias tensor for input gate. Shape [num_directions, 8*hidden_size]
                        if (ng_inputs.size() >= 4)
                        {
                            m_map["B"] = ng_inputs.at(3);
                        }
                        else
                        {
                            // XXX: single or double precision?
                            m_map["B"] = common::make_constant_node<float>(
                                    element::f32, {n_dirs, 2 * gates_count * hidden_size}, {0.f});
                        }
                        // The lengths of the sequences in a batch. Shape [batch_size]
                        if (ng_inputs.size() >= 5)
                        {
                            m_map["seq_lengths"] = ng_inputs.at(4);
                        }
                        // The initial value of the hidden. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() >= 6)
                        {
                            m_map["init_H"] = ng_inputs.at(5);
                        }
                        else
                        {
                            m_map["init_H"] = common::make_constant_node<float>(
                                        element::f32, {n_dirs, batch_size, hidden_size}, {0.f});
                        }
                        // The initial value of the cell. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() >= 7)
                        {
                            m_map["init_C"] = ng_inputs.at(6);
                        }
                        else
                        {
                            m_map["init_C"] = common::make_constant_node<float>(
                                        element::f32, {n_dirs, batch_size, hidden_size}, {0.f});
                        }
                        // The weight tensor for peepholes. Shape [num_directions, 3*hidde_size]
                        if (ng_inputs.size() >= 8)
                        {
                            m_map["P"] = ng_inputs.at(7);
                        }
                        else
                        {
                            m_map["P"] = common::make_constant_node<float>(
                                        element::f32, {n_dirs, peepholes_count * hidden_size}, {0.f});
                        }
                    }

                    NgraphNodePtr& operator[](const std::string& key)
                    {
                        return m_map[key];
                    }

                    iterator begin()
                    {
                        return m_map.begin();
                    }

                    iterator end()
                    {
                        return m_map.end();
                    }

                    std::map<std::string, NgraphNodePtr> m_map;
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                enum class LSTMDirection
                {
                    LSTM_DIRECTION_FORWARD,
                    LSTM_DIRECTION_REVERSE,
                    LSTM_DIRECTION_BIDIRECTIONAL
                };

                using ActivationFunc = std::function<NgraphNodePtr(const NgraphNodePtr&)>;
                using ActivationFuncsMap = std::unordered_map<std::string, ActivationFunc>;

                struct LSTMAttributes
                {
                    explicit LSTMAttributes(const Node& node)
                    {
                        // ---- Required -----
                        m_hidden_size = node.get_attribute_value<std::int64_t>("hidden_size");

                        // ---- Optional -----
                        m_activation_alpha = node.get_attribute_value<std::vector<float>>(
                                                    "activation_alpha", {});
                        m_activation_beta = node.get_attribute_value<std::vector<float>>(
                                                    "activation_beta", {});

                        // FIXME: causes ld errors!
                        // m_activations = node.get_attribute_value<std::vector<std::string>>(
                        //                            "activations", {"Sigmoid", "Tanh", "Tanh"});

                        // If absent - no clipping.
                        m_clip = node.get_attribute_value<float>("clip",
                                                                 {std::numeric_limits<float>::max()});
                        std::string direction =
                            node.get_attribute_value<std::string>("direction", "forward");
                        ASSERT_IS_SUPPORTED(node, (direction == "forward"))
                            << "Currently only forward mode is supported";

                        m_input_forget = static_cast<bool>(
                                            node.get_attribute_value<std::int64_t>("input_forget", 0));

                        // Register available activation functions.
                        m_atcivation_funcs.emplace("Sigmoid", std::bind(Sigmoid, std::placeholders::_1));
                        m_atcivation_funcs.emplace("Tanh", std::bind(Tanh, std::placeholders::_1));
                    }

                    std::vector<float> m_activation_alpha;
                    std::vector<float> m_activation_beta;
                    std::vector<std::string> m_activations{"Sigmoid", "Tanh", "Tanh"};
                    ActivationFuncsMap m_atcivation_funcs;
                    float m_clip;
                    LSTMDirection m_direction{LSTMDirection::LSTM_DIRECTION_FORWARD};
                    std::int64_t m_hidden_size;
                    bool m_input_forget;
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM NODE CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                class LSTMNode
                {
                public:
                    explicit LSTMNode(const Node& node)
                      : m_input_map{node},
                        m_attributes{node},
                        m_f{m_attributes.m_atcivation_funcs["Sigmoid"]},
                        m_g{m_attributes.m_atcivation_funcs["Tanh"]},
                        m_h{m_attributes.m_atcivation_funcs["Tanh"]}
                    {
                        if (m_attributes.m_direction == LSTMDirection::LSTM_DIRECTION_FORWARD)
                        {

                            // Since we have forward LSTM we can squeeze `num_directions` axis from inputs.
                            for (auto& ng_in : m_input_map)
                            {
                                if (ng_in.first != "X" && ng_in.first != "seq_lengths")
                                {
                                    ASSERT_VALID_ARGUMENT(node, ng_in.second->get_shape().at(0) == 1)
                                        << "Input: { " << ng_in.first << " } first axis has size different "
                                           "from 1, while direction attribute set to 'forward'.";
                                    ng_in.second = squeeze_first_dim(ng_in.second);
                                }
                            }
                        }
                    }
                    ~LSTMNode() {};

                    NodeVector run()
                    {
                        NodeVector p_iof = split(m_input_map["P"], 3);
                        NgraphNodePtr p_i = p_iof.at(0);
                        NgraphNodePtr p_o = p_iof.at(1);
                        NgraphNodePtr p_f = p_iof.at(2);
                        NgraphNodePtr H_t = m_input_map["init_H"];;
                        NgraphNodePtr C_t = m_input_map["init_C"];;
                        NodeVector h_list;

                        NodeVector b_W_R = split(m_input_map["B"], 2);
                        NgraphNodePtr bias = b_W_R.at(0) + b_W_R.at(1);
                        NodeVector in_seqs = split(m_input_map["X"], m_input_map["X"]->get_shape().at(0),
                                                   0, true);

                        for (const auto& in_x : in_seqs)
                        {
                            auto Xt_W = std::make_shared<ngraph::op::Dot>(in_x,
                                reshape::transpose(m_input_map["W"]));
                            auto Ht_W = std::make_shared<ngraph::op::Dot>(H_t,
                                reshape::transpose(m_input_map["R"]));
                            auto gates = add(Xt_W, add(Ht_W, bias));

                            NodeVector split_gates = split(gates, 4, -1);
                            auto i = split_gates.at(0);
                            auto o = split_gates.at(1);
                            auto f = split_gates.at(2);
                            auto c = split_gates.at(3);

                            i = m_f(add(i, mul(p_i, C_t)));
                            f = m_f(add(f, mul(p_f, C_t)));
                            auto C = add(mul(f, C_t), mul(i, m_g(c)));
                            o = m_f(add(o, mul(p_o, C)));
                            auto H = mul(o, m_h(C));
                            h_list.push_back(H);
                            H_t = H;
                            C_t = C;
                        }
                        // The tensor that concats all the intermediate output values of the hidden.
                        // It has shape [seq_length, batch_size, hidden_size]
                        NodeVector exp_h_list;
                        Shape shape{1};
                        shape.insert(std::end(shape), std::begin(H_t->get_shape()),
                                     std::end(H_t->get_shape()));
                        for (const auto& ht : h_list)
                        {
                            exp_h_list.push_back(std::make_shared<ngraph::op::Reshape>(ht,
                                    reshape::get_default_axis_vector(ht->get_shape().size()),
                                    shape));
                        }
                        NgraphNodePtr Y{std::make_shared<ngraph::op::Concat>(exp_h_list, 0)};

                        // Expand Y so that it has expected shape:
                        // [seq_length, num_directions, batch_size, hidden_size]
                        if (m_attributes.m_direction == LSTMDirection::LSTM_DIRECTION_FORWARD)
                        {
                            shape = Y->get_shape();
                            shape.insert(std::next(std::begin(shape)), 1);
                            Y = std::make_shared<ngraph::op::Reshape>(Y,
                                    reshape::get_default_axis_vector(Y->get_shape().size()),
                                    shape);
                        }
                        return {Y, exp_h_list.back()};
                    }

                private:
                    LSTMNgInputMap m_input_map;
                    LSTMAttributes m_attributes;

                    const ActivationFunc& m_f;
                    const ActivationFunc& m_g;
                    const ActivationFunc& m_h;

                    // input, output, cell, forget
                    const std::size_t m_gates_count{4};
                    // input, output, forget
                    const std::size_t m_peepholes_count{3};
                };

            } // anonymous namespace

            namespace set_1
            {
                NodeVector lstm(const Node& node)
                {
                    LSTMNode lstm{node};
                    return lstm.run();
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph