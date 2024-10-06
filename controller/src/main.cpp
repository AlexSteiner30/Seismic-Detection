#include <Arduino.h>
#include "model/model.h"
#include "dataset/dataset.h"

#include "layers/linear.h"
#include "layers/convolution.h"
#include "layers/maxpool.h"
#include "layers/flatten.h"

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

Dataset dataset;
Model model;

void setup() {
  Serial.begin(9600);
  delay(5000);  
}

void loop() {
  std::vector<std::vector<Neuron>> hidden_layer_2d = dataset.data[0];
  std::vector<Neuron> hidden_layer_1d;
  std::vector<std::vector<Neuron>> outputs;

  // Forward pass
  for (const auto &layer : model.layers){
    if (auto *flatten_layer_ptr = dynamic_cast<Flatten *>(layer.get())){
      hidden_layer_1d = flatten_layer_ptr->forward_flatten(hidden_layer_2d);
    }else if (auto *linear_layer_ptr = dynamic_cast<Linear *>(layer.get())){
      hidden_layer_1d = linear_layer_ptr->forward(hidden_layer_1d);
      if (linear_layer_ptr->is_output){
        outputs.push_back(hidden_layer_1d);
      }
    }else if (auto *conv_layer_ptr = dynamic_cast<Convolution *>(layer.get())){
      hidden_layer_2d = conv_layer_ptr->forward(hidden_layer_2d);
    }
    else if (auto *max_pool_layer_ptr = dynamic_cast<MaxPool *>(layer.get())){
      hidden_layer_2d = max_pool_layer_ptr->forward(hidden_layer_2d);
    }
  }

  std::vector<double> output_results;
  for (const auto &neuron : outputs.back()){
    output_results.push_back(neuron.output);
  }

  Serial.print("Seismic Data Detected At: ");
  Serial.println(String(std::distance(output_results.begin(), std::max_element(output_results.begin(), output_results.end())) * 7));

  delay(1000);
}
