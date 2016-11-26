#include <iostream>
#include "neural_net.h"
#include <fstream>

double TestNet(NeuralNet &net, const std::string &data_file);
void TrainNet(NeuralNet &net, const std::string &traing_file, const std::string &test_file, double epsilon, int epochs, 
              std::string save_file, int num_steps_change = -1);

int main(int argc, char **argv){
  if (argc!=4) {
    std::cout << "Incorrect usage, correct usage:\n";
    std::cout << "mnist_net_automated {instruction file} {train file} {test file}\n";
    return 0;
  } 
  std::string instruction_string(argv[1]);  
  std::string train(argv[2]);
  std::string test(argv[3]);
  std::string token;
  std::ifstream instructions(instruction_string);
  //get rid of header
  std::getline(instructions, token);
  double epsilon, sigmoid_coef;
  int epochs, num_steps_change;
  int hidden_node_number;
  std::string save_best_name;

  while(std::getline(instructions, token)){
    std::cout << "***************************************";
    std::stringstream ss(token);
    ss >> epsilon >> sigmoid_coef >> epochs >> num_steps_change >> hidden_node_number >> save_best_name;
    NeuralNet mnist_net(784, hidden_node_number, 10, sigmoid_coef);
    if(num_steps_change < 1){
      std::cout << "For fixed epsilon = " << epsilon << ", sc = " << sigmoid_coef << ", epochs = " << epochs << ",\n";
      TrainNet(mnist_net, train, test, epsilon, epochs, save_best_name, -1);
    } else {
      std::cout << "For initial epsilon = " << epsilon << " and decreasing every " << num_steps_change << " trainging instances" << ", sc = " << sigmoid_coef << ", epochs = " << epochs << ",\n";
      TrainNet(mnist_net, train, test, epsilon, epochs, save_best_name, num_steps_change);
    }
  }
  return 0;
}

void TrainNet(NeuralNet &net, const std::string &data_file, const std::string &test_file, double epsilon, int epochs, 
              std::string save_file, int num_steps_change){
  bool decreasing_learn = true;
  if(num_steps_change < 1)
    decreasing_learn = false;
  double learn_rate = epsilon;
  int counter = 0;
  int epsilon_divisor = 1;
  double best_accuracy = 0.0;
  for(int x = 0; x < epochs; ++x){
    std::ifstream in_stream(data_file);
    if(!in_stream.is_open()){
      std::cout << "Failed to open training file, exiting\n";
      exit(1);
    }
    std::string token;
    while(std::getline(in_stream, token)){
      ++counter;
      std::stringstream ss(token);
      std::vector<double> trainer;
      double k;
      int target;
      ss >> target;
      ss.ignore();
      //get and input, put in a vector
      while (ss >> k){
        trainer.push_back(k);
        //skip the comma and space
        ss.ignore();
      }
      //check to make sure input is okay
      if(trainer.size() != net.GetNumInputs() || target >= net.GetNumOutputs()){
        std::cout << "Input data formatted incorrectly, exiting.\n";
        exit(1);
      }
      for(int i = 0; i < trainer.size(); ++i)
        trainer[i] = (trainer[i] / 255)*.99 + .01;
      //set up target output vector
      //outputs are .01 and .99 as 0 and 1 are impossible outputs of sigmoid    
      std::vector<double> target_output(10, 0.01);
      target_output[target] = .99;
      net.TrainNet(trainer, target_output, learn_rate); 
      if(decreasing_learn && counter % num_steps_change == 0){
        ++epsilon_divisor;
        learn_rate = epsilon / epsilon_divisor;
      }
    }
    in_stream.close();
    double accuracy = TestNet(net, test_file);
    if(best_accuracy < accuracy){
      best_accuracy = accuracy;
      std::cout << "New best accuracy, saving to file: \"" << save_file << "\"\n";
      net.Save(save_file);
    }
    std::cout << "Accuracy at epoch " << x + 1 << " :" << accuracy << std::endl;
  }
}

double TestNet(NeuralNet &net, const std::string &data_file){
  std::ifstream in_stream(data_file);
  int counter = 0;
  if(!in_stream.is_open()){
    std::cout << "Failed to open training file, exiting\n";
    exit(1);
  }
  std::string token;
  int num_correct = 0;
  int total = 0;
  //get input vectors and update weights
  while(std::getline(in_stream, token)){
    ++counter;
    std::stringstream ss(token);
    std::vector<double> tester;
    double k;
    int target;
    ss >> target;
    ss.ignore();
    //get and input, put in a vector
    while (ss >> k){
      tester.push_back(k);
      //skip the comma and space
      ss.ignore();
    }
    //check to make sure input is okay
    if(tester.size() != net.GetNumInputs() || target >= net.GetNumOutputs()){
      std::cout << "Input data formatted incorrectly, exiting.\n";
      exit(1);
    }
    std::vector<double> output = net.QueryNet(tester);
    int max_position = 0;
    double max_value = output[0];
    for(int i = 1; i < 10; ++i){
      if(max_value < output[i]){
        max_position = i;
        max_value = output[i];
      }
    }
    if(max_position == target)
      ++num_correct;
    ++total;    
  }
  in_stream.close();
  double percent_correct = ((double) num_correct) / total;
  return percent_correct;
}
