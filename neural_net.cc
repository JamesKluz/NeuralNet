//Implementation file for neural_net.h
//Author: James Kluz

#include "neural_net.h"
#include </usr/local/include/Eigen/Dense>
#include <iostream>
#include <iomanip>  
#include <random>
#include <cmath>
#include <math.h>  
#include <fstream>
#include <vector>
#include <sstream>

using namespace Eigen;

NeuralNet::NeuralNet(int inputs, int hidden_nodes, int outputs, double s_coef, bool norm_on_correct){
  num_inputs_ = inputs;
  num_outputs_ = outputs;
  num_hidden_ = hidden_nodes;
  normalize_on_correction_ = norm_on_correct;
  sigmoid_coefficient_ = s_coef;
  weights_input_hidden_ = MatrixXd(hidden_nodes, inputs);
  weights_hidden_output_  = MatrixXd(outputs, hidden_nodes);
  //Initialize weights to a normal distribution centered at 0 with SD = 1/sqroot(incoming links)
  std::random_device rd;   //random device
  std::mt19937 gen(rd());  //twister_engine -> high quality random numbers
  double input_to_hidden_sigma = pow(inputs, -0.5); //1/sqroot(incoming links)
  double hidden_to_output_sigma = pow(hidden_nodes, -0.5);
  std::normal_distribution<double> d_1(0.0, input_to_hidden_sigma);
  std::normal_distribution<double> d_2(0.0, hidden_to_output_sigma);
  //initialize input to hidden weight matrix
  double scalar;
  for(int i = 0; i < weights_input_hidden_.rows(); ++i)
    for(int j = 0; j < weights_input_hidden_.cols(); ++j){
      scalar = d_1(gen);
      //don't want any 0.0 initial weights
      while(scalar == 0.0) 
        scalar = d_1(gen);
      weights_input_hidden_(i, j) = scalar;
    }
  //initialize hidden to output weight matrix
  for(int i = 0; i < weights_hidden_output_.rows(); ++i)
    for(int j = 0; j < weights_hidden_output_.cols(); ++j){
      scalar = d_2(gen);
      //don't want any 0.0 initial weights
      while(scalar == 0.0) 
        scalar = d_2(gen);
      weights_hidden_output_(i, j) = scalar;
    } 
}

NeuralNet::NeuralNet(std::string loader){
  Load(loader); 
}

std::vector<double> NeuralNet::QueryNet(std::vector<double> &query_input) const {
  if(query_input.size() != num_inputs_){
    std::cout << "input vector should be size " << num_inputs_ << ". Exiting.\n";
    exit(1);
  }
  //Covert input std::vector to an Eigen::Vector
  double* data = &query_input[0];
  Eigen::Map<Eigen::VectorXd> input_vector(data, num_inputs_);
  MatrixXd hidden_vector(num_hidden_, 1);
  MatrixXd output_vector(num_outputs_, 1);
  hidden_vector = weights_input_hidden_ * input_vector;
  for(int i = 0; i < num_hidden_; ++i)
    hidden_vector(i, 0) = Sigmoid(hidden_vector(i, 0));  
  output_vector = weights_hidden_output_ * hidden_vector;
  std::vector<double> output_for_return(num_outputs_, 0.0);
  for(int i = 0; i < num_outputs_; ++i)
    output_for_return[i] = Sigmoid(output_vector(i, 0));  
  return output_for_return;
}

void NeuralNet::TrainNet(std::vector<double> &train_input, std::vector<double> &target_ouput, 
                                        double learn_rate){
  if(train_input.size() != num_inputs_){
    std::cout << "input vector should be size " << num_inputs_ << ". Exiting.\n";
    exit(1);
  }
  //Covert input std::vector to an Eigen::Vector
  double* data = &train_input[0];
  Eigen::Map<Eigen::VectorXd> input_vector(data, num_inputs_);
  MatrixXd hidden_vector(num_hidden_, 1);
  MatrixXd output_vector(num_outputs_, 1);
  hidden_vector = weights_input_hidden_ * input_vector;
  for(int i = 0; i < num_hidden_; ++i)
    hidden_vector(i, 0) = Sigmoid(hidden_vector(i, 0));  
  output_vector = weights_hidden_output_ * hidden_vector;
  for(int i = 0; i < num_outputs_; ++i)
    output_vector(i, 0) = Sigmoid(output_vector(i, 0)); 
  MatrixXd error_vector(num_outputs_, 1);   
  for(int i = 0; i < num_outputs_; ++i)
    error_vector(i, 0) = target_ouput[i] - output_vector(i, 0);
  MatrixXd hidden_error(num_hidden_, 1);
  //check to see if we are going to normalize the transpose
  if(normalize_on_correction_){
    MatrixXd weights_hidden_output_normed(num_outputs_, num_hidden_);
    //vector to store sums of rows of weight matrix
    std::vector<double> sums(num_outputs_ , 0.0);
    for(int i = 0; i < num_outputs_; ++i)
      for(int j = 0; j < num_hidden_; ++j)
        sums[i] += weights_hidden_output_(i, j);
    //normalize the weight matrix
    for(int i = 0; i < num_outputs_; ++i)
      for(int j = 0; j < num_hidden_; ++j)
        weights_hidden_output_normed(i, j) = weights_hidden_output_(i, j) / sums[i];
    hidden_error = weights_hidden_output_normed.transpose() * error_vector;      
  } else {
    hidden_error = weights_hidden_output_.transpose() * error_vector;
  }
  //now we correct the weights
  //vector for component-wise multiplication and addition:
  //want temp[k] = 2 * learning_rate * Sigmoid_coefficient * error[i]*actual[k]*(1 - actual[i]) 
  MatrixXd temp_1(num_outputs_, 1);
  for(int i = 0; i < num_outputs_; ++i)
    temp_1(i, 0) = 2 * learn_rate * error_vector(i, 0) * output_vector(i, 0) * (1 - output_vector(i, 0));
  //correct hidden to output weights
  weights_hidden_output_ = weights_hidden_output_ + temp_1 * hidden_vector.transpose();
  //vector for component-wise multiplication and addition:
  MatrixXd temp_2(num_hidden_, 1);
  for(int i = 0; i < num_hidden_; ++i)
    temp_2(i, 0) = 2 * learn_rate * hidden_error(i, 0) * hidden_vector(i, 0) * (1 - hidden_vector(i, 0));
  //correct hidden to output weights
  weights_input_hidden_ = weights_input_hidden_ + temp_2 * input_vector.transpose();
}

void NeuralNet::Save(std::string save_file) const{
  std::ofstream out_stream(save_file);
  if(!out_stream.is_open()){
    std::cout << "Failed to open save file, exiting\n";
    exit(1);
  }
  out_stream << num_hidden_ << " " << num_inputs_ << " " << num_outputs_ << " " << sigmoid_coefficient_ << " " << normalize_on_correction_  << "\n";
  for(int i = 0; i < num_hidden_; ++i)
    for(int j = 0; j < num_inputs_; ++j)
      out_stream << weights_input_hidden_(i, j) << " ";
  out_stream << "\n";
  for(int i = 0; i < num_outputs_; ++i)
    for(int j = 0; j < num_hidden_; ++j)
      out_stream << weights_hidden_output_(i, j) << " ";
  out_stream << "\n";
  out_stream.close();
}

void NeuralNet::Load(std::string load_file){
  std::ifstream in_stream(load_file);
  if(!in_stream.is_open()){
    std::cout << "Failed to open load file, exiting\n";
    exit(1);
  }
  std::string token;
  std::getline(in_stream, token);
  std::stringstream ss_dimensions(token);
  ss_dimensions >> num_hidden_ >> num_inputs_ >> num_outputs_ >> sigmoid_coefficient_ >> normalize_on_correction_;
  weights_input_hidden_ = MatrixXd(num_hidden_, num_inputs_);
  weights_hidden_output_  = MatrixXd(num_outputs_, num_hidden_);
  std::getline(in_stream, token);
  std::stringstream ss_scalars_1(token);
  for(int i = 0; i < num_hidden_; ++i)
    for(int j = 0; j < num_inputs_; ++j)
      ss_scalars_1 >> weights_input_hidden_(i, j);
  std::getline(in_stream, token);
  std::stringstream ss_scalars_2(token);  
  for(int i = 0; i < num_outputs_; ++i)
    for(int j = 0; j < num_hidden_; ++j)
      ss_scalars_2 >> weights_hidden_output_(i, j);
  in_stream.close();
}

double NeuralNet::Sigmoid(double x) const{
  return 1/(1 + exp(sigmoid_coefficient_ * (-x)));
}